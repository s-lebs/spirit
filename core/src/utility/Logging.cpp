#include <io/IO.hpp>
#include <utility/Logging.hpp>
#include <utility/Timing.hpp>

#include <fmt/format.h>

#include <termcolor/termcolor.hpp>

#include <signal.h>
#include <ctime>
#include <iostream>
#include <string>

namespace Utility
{

std::string SenderToString( Log_Sender sender, bool braces_separators = true )
try
{
    std::string result = "";

    // Braces
    if( braces_separators )
        result.append( "[" );
    else
        result.append( " " );
    // Sender
    if( sender == Log_Sender::All )
        result.append( "ALL " );
    else if( sender == Log_Sender::IO )
        result.append( "IO  " );
    else if( sender == Log_Sender::API )
        result.append( "API " );
    else if( sender == Log_Sender::GNEB )
        result.append( "GNEB" );
    else if( sender == Log_Sender::HTST )
        result.append( "HTST" );
    else if( sender == Log_Sender::LLG )
        result.append( "LLG " );
    else if( sender == Log_Sender::MC )
        result.append( "MC  " );
    else if( sender == Log_Sender::MMF )
        result.append( "MMF " );
    else if( sender == Log_Sender::UI )
        result.append( "UI  " );
    else if( sender == Log_Sender::EMA )
        result.append( "EMA " );
    // Braces
    if( braces_separators )
        result.append( "]" );
    else
        result.append( " " );

    return result;
}
catch( ... )
{
    spirit_rethrow( "Could not convert sender to string" );
    return "";
}

std::string LevelToString( Log_Level level, bool braces_separators = true )
try
{
    std::string result = "";

    // Braces
    if( braces_separators )
        result.append( "[" );
    else
        result.append( " " );
    // Level
    if( level == Log_Level::All )
        result.append( "  ALL  " );
    else if( level == Log_Level::Severe )
        result.append( "SEVERE " );
    else if( level == Log_Level::Error )
        result.append( " ERROR " );
    else if( level == Log_Level::Warning )
        result.append( "WARNING" );
    else if( level == Log_Level::Parameter )
        result.append( " PARAM " );
    else if( level == Log_Level::Info )
        result.append( " INFO  " );
    else if( level == Log_Level::Debug )
        result.append( " DEBUG " );
    // Braces
    if( braces_separators )
        result.append( "]" );
    else
        result.append( " " );

    return result;
}
catch( ... )
{
    spirit_rethrow( "Could not convert log level to string" );
    return "";
}

std::string IndexToString( int idx, bool braces_separators = true )
try
{
    std::string result = "";

    // Braces
    if( braces_separators )
        result.append( "[" );
    else
        result.append( " " );
    // Index
    std::string idx_str = fmt::format( "{:0>2}", idx + 1 );
    if( idx >= 0 )
        result.append( idx_str );
    else
        result.append( "--" );
    // Braces
    if( braces_separators )
        result.append( "]" );
    else
        result.append( " " );

    return result;
}
catch( ... )
{
    spirit_rethrow( "Could not convert index integer to string" );
    return "";
}

std::string LogEntryToString( LogEntry entry, bool braces_separators )
try
{
    std::string result = "";
    // Time
    result += Timing::TimePointToString_Pretty( entry.time );
    // Message Level
    result += "  " + LevelToString( entry.level, braces_separators );
    // Sender
    result += " " + SenderToString( entry.sender, braces_separators );
    // Chain Index
    result += " " + IndexToString( entry.idx_chain, braces_separators );
    // Image Index
    result += " " + IndexToString( entry.idx_image, braces_separators );
    // First message string
    result += "  " + entry.message_lines[0];
    // Rest of the block
    for( unsigned int i = 1; i < entry.message_lines.size(); ++i )
        result += "\n" + Log.tags_space + entry.message_lines[i];
    // Return
    return result;
}
catch( ... )
{
    spirit_rethrow( "Could not convert log entry to string" );
    return "";
}

LoggingHandler::LoggingHandler()
{
    // Set the default Log parameters
    output_folder = ".";

    if( file_tag == std::string( "<time>" ) )
        fileName = "Log_" + Utility::Timing::CurrentDateTime() + ".txt";
    else if( file_tag == std::string( "" ) )
        fileName = "Log.txt";
    else
        fileName = "Log_" + file_tag + ".txt";

    messages_to_file        = false;
    level_file              = Log_Level::Debug;
    messages_to_console     = true;
    level_console           = Log_Level::Parameter;
    save_input_initial      = true;
    save_input_final        = false;
    save_positions_initial  = false;
    save_positions_final    = false;
    save_neighbours_initial = false;
    save_neighbours_final   = false;
    n_entries               = 0;
    n_errors                = 0;
    n_warnings              = 0;
    n_dumped                = 0;
}

void LoggingHandler::Send(
    Log_Level level, Log_Sender sender, const std::string & message, int idx_image, int idx_chain )
try
{
    // Lock mutex because of reallocation (push_back)
    std::lock_guard<std::mutex> guard( mutex );

    // All messages are saved in the Log
    LogEntry entry = { std::chrono::system_clock::now(), sender, level, { message }, idx_image, idx_chain };
    log_entries.push_back( entry );

    // Increment message count
    n_entries++;
    // Increment error count
    if( level == Log_Level::Error )
        n_errors++;
    // Increment warning count
    if( level == Log_Level::Warning )
        n_warnings++;

    // Determine message color in console
    auto color = termcolor::reset;
    if( level <= Log_Level::Warning )
        color = termcolor::yellow;
    if( level <= Log_Level::Error )
        color = termcolor::red;
    if( level == Log_Level::All )
        color = termcolor::reset;

    // If level <= verbosity, we print to console, but Error and Severe are always printed
    if( ( messages_to_console && level <= level_console ) || ( level == Log_Level::Error )
        || ( level == Log_Level::Severe ) )
        std::cout << color << LogEntryToString( log_entries.back() ) << termcolor::reset << "\n";
}
catch( ... )
{
    spirit_rethrow( "LoggingHandler::Send failed" );
}

void LoggingHandler::SendBlock(
    Log_Level level, Log_Sender sender, const std::vector<std::string> & messages, int idx_image, int idx_chain )
try
{
    // Lock mutex because of reallocation (push_back)
    std::lock_guard<std::mutex> guard( mutex );

    // All messages are saved in the Log
    LogEntry entry{ std::chrono::system_clock::now(), sender, level, messages, idx_image, idx_chain };
    log_entries.push_back( entry );

    // Increment message count
    n_entries++;

    // Determine message color in console
    auto color = termcolor::reset;
    if( level <= Log_Level::Warning )
        color = termcolor::yellow;
    if( level <= Log_Level::Error )
        color = termcolor::red;
    if( level == Log_Level::All )
        color = termcolor::reset;

    // If level <= verbosity, we print to console, but Error and Severe are always printed
    if( ( messages_to_console && level <= level_console ) || level == Log_Level::Error || level == Log_Level::Severe )
        std::cout << color << LogEntryToString( log_entries.back() ) << termcolor::reset << "\n";
}
catch( ... )
{
    spirit_rethrow( "LoggingHandler::SendBlock failed" );
}

void LoggingHandler::operator()(
    Log_Level level, Log_Sender sender, const std::string & message, int idx_image, int idx_chain )
try
{
    Send( level, sender, message, idx_image, idx_chain );
}
catch( ... )
{
    spirit_rethrow( "Log(...) call failed" );
}

void LoggingHandler::operator()(
    Log_Level level, Log_Sender sender, const std::vector<std::string> & messages, int idx_image, int idx_chain )
try
{
    SendBlock( level, sender, messages, idx_image, idx_chain );
}
catch( ... )
{
    spirit_rethrow( "Log(...) call failed" );
}

std::vector<LogEntry> LoggingHandler::GetEntries()
try
{
    return log_entries;
}
catch( ... )
{
    spirit_rethrow( "Unable to retrieve log entries" );
    return {};
}

std::vector<LogEntry> LoggingHandler::Filter( Log_Level level, Log_Sender sender, int idx_image, int idx_chain )
try
{
    // Get vector of Log entries
    std::vector<LogEntry> result( 0 );
    for( const auto & entry : log_entries )
    {
        if( ( level == Log_Level::All || level == entry.level )
            && ( sender == Log_Sender::All || sender == entry.sender )
            && ( idx_image == -1 || idx_image == entry.idx_image )
            && ( idx_chain == -1 || idx_chain == entry.idx_chain ) )
        {
            result.push_back( entry );
        }
    }

    // Return
    return std::vector<LogEntry>();
}
catch( ... )
{
    spirit_rethrow( "Error filtering the log" );
    return {};
}

void LoggingHandler::Append_to_File()
try
{
    if( this->messages_to_file )
    {
        // Log this event
        Send(
            Log_Level::Debug, Log_Sender::All,
            fmt::format( "Appending log to file \"{}/{}\"", output_folder, fileName ) );

        // Gather the string
        std::string logstring = "";
        int begin_append      = n_dumped;
        n_dumped              = n_entries;
        for( int i = begin_append; i < n_entries; ++i )
        {
            auto level = log_entries[i].level;
            if( level <= level_file || level == Log_Level::Error || level == Log_Level::Severe )
            {
                logstring.append( LogEntryToString( log_entries[i] ) );
                logstring.append( "\n" );
            }
        }

        // Append to file
        IO::Append_String_to_File( logstring, output_folder + "/" + fileName );
    }
    else
    {
        Send(
            Log_Level::Debug, Log_Sender::All,
            fmt::format( "Not appending log to file \"{}/{}\"", output_folder, fileName ) );
    }
}
catch( ... )
{
    spirit_rethrow( "Could not append log to file" );
}

// Write the entire Log to file
void LoggingHandler::Dump_to_File()
try
{
    if( this->messages_to_file )
    {
        // Log this event
        Send(
            Log_Level::Info, Log_Sender::All, fmt::format( "Dumping log to file \"{}/{}\"", output_folder, fileName ) );

        // Gather the string
        std::string logstring = "";
        for( int i = 0; i < n_entries; ++i )
        {
            auto level = log_entries[i].level;
            if( level <= level_file || level == Log_Level::Error || level == Log_Level::Severe )
            {
                logstring.append( LogEntryToString( log_entries[i] ) );
                logstring.append( "\n" );
            }
        }

        // Write the string to file
        IO::String_to_File( logstring, output_folder + "/" + fileName );
    }
    else
    {
        Send(
            Log_Level::Debug, Log_Sender::All,
            fmt::format( "Not dumping log to file \"{}/{}\"", output_folder, fileName ) );
    }
}
catch( ... )
{
    spirit_rethrow( "Could not dump log to file" );
}

} // end namespace Utility
#pragma once
#ifndef INTERFACE_LOG_H
#define INTERFACE_LOG_H
#include "DLL_Define_Export.h"

#include <vector>

// Define Log Levels
#define Log_Level_All       0
#define Log_Level_Severe    1
#define Log_Level_Error     2
#define Log_Level_Warning   3
#define Log_Level_Parameter 4
#define Log_Level_Info      5
#define Log_Level_Debug     6

// Define Log Senders
#define Log_Sender_All  0
#define Log_Sender_IO   1
#define Log_Sender_GNEB 2
#define Log_Sender_LLG  3
#define Log_Sender_MC   4
#define Log_Sender_MMF  5
#define Log_Sender_API  6
#define Log_Sender_UI   7

struct State;

//      General functions
// Send a Log message
DLLEXPORT void Log_Send(State *state, int level, int sender, const char * message, int idx_image=-1, int idx_chain=-1) noexcept;
// Get the entries from the Log and write new number of entries into given int
// TODO: can this be written in a C-style way?
namespace Utility
{
    struct LogEntry;
}
std::vector<Utility::LogEntry> Log_Get_Entries(State *state) noexcept;
// Append the Log to it's file
DLLEXPORT void Log_Append(State *state) noexcept;
// Dump the Log into it's file
DLLEXPORT void Log_Dump(State *state) noexcept;
// Get the number of Log entries
DLLEXPORT int Log_Get_N_Entries(State *state) noexcept;
// Get the number of errors in the Log
DLLEXPORT int Log_Get_N_Errors(State *state) noexcept;
// Get the number of warnings in the Log
DLLEXPORT int Log_Get_N_Warnings(State *state) noexcept;

//      Set Log parameters
DLLEXPORT void Log_Set_Output_File_Tag(State *state, const char * tag) noexcept;
DLLEXPORT void Log_Set_Output_Folder(State *state, const char * folder) noexcept;
DLLEXPORT void Log_Set_Output_To_Console(State *state, bool b) noexcept;
DLLEXPORT void Log_Set_Output_Console_Level(State *state, int level) noexcept;
DLLEXPORT void Log_Set_Output_To_File(State *state, bool b) noexcept;
DLLEXPORT void Log_Set_Output_File_Level(State *state, int level) noexcept;

//      Get Log parameters
DLLEXPORT const char * Log_Get_Output_File_Tag(State *state) noexcept;
DLLEXPORT const char * Log_Get_Output_Folder(State *state) noexcept;
DLLEXPORT bool Log_Get_Output_To_Console(State *state) noexcept;
DLLEXPORT int Log_Get_Output_Console_Level(State *state) noexcept;
DLLEXPORT bool Log_Get_Output_To_File(State *state) noexcept;
DLLEXPORT int Log_Get_Output_File_Level(State *state) noexcept;

#include "DLL_Undefine_Export.h"
#endif
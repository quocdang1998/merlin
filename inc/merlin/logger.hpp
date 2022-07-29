// Copyright 2022 quocdang1998
#ifndef MERLIN_LOGGER_HPP_
#define MERLIN_LOGGER_HPP_

#include <cstdio>
#include <string>

// define log messages for GNU g++ on Linux
#if defined(__GNUG__)
#define MESSAGE(fmt, ...) logger_(LogMode::Message, __PRETTY_FUNCTION__, fmt, ## __VA_ARGS__)
#define WARNING(fmt, ...) logger_(LogMode::Warning, __PRETTY_FUNCTION__, fmt, ##__VA_ARGS__)
#define FAILURE(fmt, ...) logger_(LogMode::Failure, __PRETTY_FUNCTION__, fmt, ##__VA_ARGS__)

// define log messages for MSVC cl.exe on Windows
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define MESSAGE(fmt, ...) logger_(LogMode::Message, __FUNCSIG__, fmt, ##__VA_ARGS__)
#define WARNING(fmt, ...) logger_(LogMode::Warning, __FUNCSIG__, fmt, ##__VA_ARGS__)
#define FAILURE(fmt, ...) logger_(LogMode::Failure, __FUNCSIG__, fmt, ##__VA_ARGS__)
#endif

enum class LogMode {
    Message,
    Warning,
    Failure
};

void logger_(LogMode mode, const std::string & func_name, const char * fmt, ...);

#endif  // MERLIN_LOGGER_HPP_

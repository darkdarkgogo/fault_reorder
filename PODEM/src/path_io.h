#ifndef ATPG_PATH_IO_H
#define ATPG_PATH_IO_H

#include <fstream>
#include <string>
#ifdef _MSC_VER
#include <codecvt>
#include <locale>
#endif

// pybind11 supplies UTF-8 paths. MSVC's narrow streams use the Windows code
// page, so use its wide-path overload; retain native CLI paths as a fallback.
inline std::ifstream open_atpg_input(const std::string &path,
                                    std::ios::openmode mode = std::ios::in) {
#ifdef _MSC_VER
  try {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::ifstream stream(converter.from_bytes(path), mode);
    if (stream) return stream;
  } catch (const std::range_error &) {
    // Older command-line callers may pass the active Windows code page.
  }
#endif
  return std::ifstream(path, mode);
}

#endif

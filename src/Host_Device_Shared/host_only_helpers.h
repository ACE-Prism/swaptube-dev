// Helpers that cannot be shared with device code, because they need <string> or
// printf. Metal forbids the C++ standard library, so these were split out of
// helpers.h to keep that header reachable from src/Metal/*.metal.

#pragma once
#include <cstdio>
#include <string>
#include "vec.h"

inline bool is_single_letter(const std::string& str) { return str.length() == 1 && isalpha(str[0]); }
inline void print_vec2(vec2 v) { printf("vec2(%.3f, %.3f)\n", v.x, v.y); }
inline void print_vec3(vec3 v) { printf("vec3(%.3f, %.3f, %.3f)\n", v.x, v.y, v.z); }
inline void print_vec4(vec4 v) { printf("vec4(%.3f, %.3f, %.3f, %.3f)\n", v.x, v.y, v.z, v.w); }

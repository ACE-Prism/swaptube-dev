// Everything in this file is inline, c-compatible, and therefore usable in both CUDA and C++.
// It is also reached from Metal, so it must stay free of the C++ standard
// library. Host-only helpers that need <string> or printf live in
// host_only_helpers.h instead.

#pragma once
#include "vec.h"
#ifndef __METAL_VERSION__ // Metal forbids the C++ standard library
#include <cstdlib>
#include "math.h"
#include <cmath>
using namespace std;
#endif
#include "shared_precompiler_directives.h"

SHARED_FILE_PREFIX

HOST_DEVICE inline float clamp(float val, float bottom, float top){return min(top, max(val, bottom));}
HOST_DEVICE inline float square(float x){return x * x;}
HOST_DEVICE inline float cube(float x){return x * x * x;}
HOST_DEVICE inline float fourth(float x){return square(square(x));}
HOST_DEVICE inline float smoother1(float x){return 3*x*x-2*x*x*x;} // We used to use this but not anymore
HOST_DEVICE inline float smoother2(float x){return x<.5 ? square(x)*2 : 1-square(1-x)*2;}
HOST_DEVICE inline float lerp(float a, float b, float w){return a*(1-w)+b*w;}
HOST_DEVICE inline float float_lerp(float a, float b, float w){return a*(1-w)+b*w;}
HOST_DEVICE inline vec2 veclerp(vec2 a, vec2 b, float w){return a*(1-w)+b*w;}
HOST_DEVICE inline vec3 veclerp(vec3 a, vec3 b, float w){return a*(1-w)+b*w;}
HOST_DEVICE inline vec4 veclerp(vec4 a, vec4 b, float w){return a*(1-w)+b*w;}
HOST_DEVICE inline float smoothlerp(float a, float b, float w){float v = smoother2(w);return a*(1-v)+b*v;}
HOST_DEVICE inline float geom_mean(float x, float y) { return sqrt(x*y); }
HOST_DEVICE inline int signum(float x) { return (x > 0) - (x < 0); }

// Returns a non-negative remainder of a divided by b, even if a is negative.
HOST_DEVICE inline float extended_mod(float a, float b) {
    b = fabs(b);
    float result = fmod(a, b);
    if (result < 0) {
        result += b;  // Ensures non-negative remainder
    }
    return result;
}

HOST_DEVICE inline float bezier(const float point1, const float point2, const float point3, const float point4, const float t) {
    float u = 1.0f - t;
    float tt = t * t;
    float uu = u * u;
    float uuu = uu * u;
    float ttt = tt * t;

    return uuu * point1 + 3.0f * uu * t * point2 + 3.0f * u * tt * point3 + ttt * point4;
}

HOST_DEVICE inline vec2 bezier_2d(const vec2& point1, const vec2& point2, const vec2& point3, const vec2& point4, const float t) {
    float u = 1.0f - t;
    float tt = t * t;
    float uu = u * u;
    float uuu = uu * u;
    float ttt = tt * t;

    return uuu * point1 + 3.0f * uu * t * point2 + 3.0f * u * tt * point3 + ttt * point4;
}

HOST_DEVICE inline vec2 pixel_to_point_in_screen(const vec2& pixel, const vec2& lx_ty, const vec2& rx_by, const vec2& wh) {
    const vec2 flip(pixel.x, wh.y-1-pixel.y);
    return flip * (rx_by - lx_ty) / wh + lx_ty;
}

HOST_DEVICE inline vec2 point_to_pixel_in_screen(const vec2& point, const vec2& lx_ty, const vec2& rx_by, const vec2& wh) {
    const vec2 flip((point - lx_ty) * wh / (rx_by - lx_ty));
    return vec2(flip.x, wh.y-1-flip.y);
}

SHARED_FILE_SUFFIX

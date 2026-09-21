#pragma once

#ifndef __METAL_VERSION__ // Metal forbids the C++ standard library
#include <cstdint>
#endif
#include "../Host_Device_Shared/helpers.h"

// Colors are everywhere. For the sake of speed, we do not give them a dedicated class.
// They are ints under the hood, and are always 32-bit, 4-channel ARGB.

SHARED_FILE_PREFIX

HOST_DEVICE inline uint32_t argb(int a, int r, int g, int b){return (a<<24) | (r<<16) | (g<<8) | b;}
HOST_DEVICE inline int geta(int col){return (col&0xff000000)>>24;}
HOST_DEVICE inline int getr(int col){return (col&0x00ff0000)>>16;}
HOST_DEVICE inline int getg(int col){return (col&0x0000ff00)>>8 ;}
HOST_DEVICE inline int getb(int col){return (col&0x000000ff)    ;}

HOST_DEVICE inline uint32_t colorlerp(int col1, int col2, float w){return argb(round(lerp(geta(col1), geta(col2), w)),
                                                                      round(lerp(getr(col1), getr(col2), w)),
                                                                      round(lerp(getg(col1), getg(col2), w)),
                                                                      round(lerp(getb(col1), getb(col2), w)));}

HOST_DEVICE inline uint32_t color_combine(int base_color, int over_color, float overlay_opacity_multiplier = 1) {
    float base_opacity = geta(base_color) / 255.0;
    float over_opacity = geta(over_color) / 255.0 * overlay_opacity_multiplier;
    float final_opacity = 1 - (1 - base_opacity) * (1 - over_opacity);
    if (final_opacity == 0) return 0x00000000;
    int final_alpha = round(final_opacity * 255.0);
    float chroma_weight = over_opacity / final_opacity;
    int final_rgb = colorlerp(base_color, over_color, chroma_weight) & 0x00ffffff;
    return (final_alpha << 24) | (final_rgb);
}

// These take float rather than double because Metal has no double at all. The
// extra precision was never meaningful here: every one of them ends up quantized
// to 8 bits per channel.
HOST_DEVICE inline uint32_t black_to_blue_to_white(float w){
    int rainbow_part1 = max(0.0f,min(1.0f,w*2-0))*255.0f;
    int rainbow_part2 = max(0.0f,min(1.0f,w*2-1))*255.0f;
    return argb(255, rainbow_part2, rainbow_part2, rainbow_part1);
}

// Convert HSV to RGB
// h, s, v are in the range [0, 1]
HOST_DEVICE inline uint32_t HSVtoRGB(float h, float s, float v, int alpha = 255) {
    float r_f, g_f, b_f;

    if (s == 0.0f) {
        // Achromatic (grey)
        r_f = g_f = b_f = v;
    } else {
        h = fmod(h, 1.0f) * 6.0f;  // Hue sector [0, 6)
        int i = h;
        float f = h - i;
        float p = v * (1.0f - s);
        float q = v * (1.0f - s * f);
        float t = v * (1.0f - s * (1.0f - f));

        switch (i) {
            case 0: r_f = v; g_f = t; b_f = p; break;
            case 1: r_f = q; g_f = v; b_f = p; break;
            case 2: r_f = p; g_f = v; b_f = t; break;
            case 3: r_f = p; g_f = q; b_f = v; break;
            case 4: r_f = t; g_f = p; b_f = v; break;
            case 5: default: r_f = v; g_f = p; b_f = q; break;
        }
    }

    // Scale to [0, 255] and clamp
    int r = clamp(static_cast<int>(round(r_f * 255.0f)), 0, 255);
    int g = clamp(static_cast<int>(round(g_f * 255.0f)), 0, 255);
    int b = clamp(static_cast<int>(round(b_f * 255.0f)), 0, 255);
    return argb(alpha, r, g, b);
}

HOST_DEVICE inline uint32_t pendulum_color(float angle1, float angle2, float p1, float p2) {
    float sa1 = sin(angle1) + 0.000001f;
    float sa2 = sin(angle2);
    float h = atan2(sa2, sa1)/6.283f+1;
    float s = min((square(sa1) + square(sa2))*5.0f,1.0f);
    float v = 1-min(.1f * sqrt(p1*p1+p2*p2), 1.0f);
    return HSVtoRGB(h, s, v);
}

HOST_DEVICE inline float linear_srgb_to_srgb(float x) {
    if (x >= 0.0031308f)
        return 1.055f*pow(x, 1.0f/2.4f) - 0.055f;
    return 12.92f * x;
}

HOST_DEVICE inline uint32_t OKLABtoRGB(int alpha, float L, float a, float b)
{
    float l_ = L + 0.3963377774f * a + 0.2158037573f * b;
    float m_ = L - 0.1055613458f * a - 0.0638541728f * b;
    float s_ = L - 0.0894841775f * a - 1.2914855480f * b;

    float l = l_*l_*l_;
    float m = m_*m_*m_;
    float s = s_*s_*s_;

    return argb(
        clamp(alpha, 0, 255),
        clamp(static_cast<int>(round(255*linear_srgb_to_srgb(+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s))), 0, 255),
        clamp(static_cast<int>(round(255*linear_srgb_to_srgb(-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s))), 0, 255),
        clamp(static_cast<int>(round(255*linear_srgb_to_srgb(-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s))), 0, 255)
    );
}

HOST_DEVICE inline uint32_t rainbow(float x, int alpha = 255, float lightness = 0.8, float saturation = 0.25){
    float angle = x * 6.2831853071796;
    return OKLABtoRGB(alpha, lightness, saturation*sin(angle), saturation*cos(angle));
}

SHARED_FILE_SUFFIX

// CPU stand-in for thrust::complex.
//
// The GPU code uses thrust for exactly one thing: thrust::complex<float> in
// color.cuh, complex_functions.cuh, manifold.cu, real_function.cu and
// two_d_algebra.cu. std::complex is not a drop-in replacement because the call
// sites mix in double and int scalars, which std::complex's operators reject.
#pragma once
#include <cmath>

namespace thrust {

template <class T>
struct complex {
    T re, im;

    complex() : re(0), im(0) {}
    template <class U> complex(U r) : re(static_cast<T>(r)), im(0) {}
    template <class U, class V> complex(U r, V i) : re(static_cast<T>(r)), im(static_cast<T>(i)) {}

    T real() const { return re; }
    T imag() const { return im; }
    void real(T r) { re = r; }
    void imag(T i) { im = i; }

    complex& operator+=(const complex& o) { re += o.re; im += o.im; return *this; }
    complex& operator-=(const complex& o) { re -= o.re; im -= o.im; return *this; }
    complex& operator*=(const complex& o) {
        const T r = re * o.re - im * o.im;
        im = re * o.im + im * o.re;
        re = r;
        return *this;
    }
};

template <class T> complex<T> operator+(const complex<T>& a) { return a; }
template <class T> complex<T> operator-(const complex<T>& a) { return complex<T>(-a.re, -a.im); }

template <class T> complex<T> operator+(const complex<T>& a, const complex<T>& b) { return complex<T>(a.re + b.re, a.im + b.im); }
template <class T> complex<T> operator-(const complex<T>& a, const complex<T>& b) { return complex<T>(a.re - b.re, a.im - b.im); }
template <class T> complex<T> operator*(const complex<T>& a, const complex<T>& b) {
    return complex<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
template <class T> complex<T> operator/(const complex<T>& a, const complex<T>& b) {
    const T d = b.re * b.re + b.im * b.im;
    return complex<T>((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
}

// Mixed scalar arithmetic: the call sites pass float, double and int literals.
template <class T, class S> complex<T> operator*(const complex<T>& a, S s) { return complex<T>(a.re * static_cast<T>(s), a.im * static_cast<T>(s)); }
template <class T, class S> complex<T> operator*(S s, const complex<T>& a) { return complex<T>(a.re * static_cast<T>(s), a.im * static_cast<T>(s)); }
template <class T, class S> complex<T> operator/(const complex<T>& a, S s) { return complex<T>(a.re / static_cast<T>(s), a.im / static_cast<T>(s)); }
template <class T, class S> complex<T> operator/(S s, const complex<T>& a) { return complex<T>(static_cast<T>(s)) / a; }
template <class T, class S> complex<T> operator+(const complex<T>& a, S s) { return complex<T>(a.re + static_cast<T>(s), a.im); }
template <class T, class S> complex<T> operator+(S s, const complex<T>& a) { return complex<T>(a.re + static_cast<T>(s), a.im); }
template <class T, class S> complex<T> operator-(const complex<T>& a, S s) { return complex<T>(a.re - static_cast<T>(s), a.im); }
template <class T, class S> complex<T> operator-(S s, const complex<T>& a) { return complex<T>(static_cast<T>(s) - a.re, -a.im); }

template <class T> bool operator==(const complex<T>& a, const complex<T>& b) { return a.re == b.re && a.im == b.im; }
template <class T> bool operator!=(const complex<T>& a, const complex<T>& b) { return !(a == b); }

template <class T> T abs(const complex<T>& a) { return std::sqrt(a.re * a.re + a.im * a.im); }
template <class T> T norm(const complex<T>& a) { return a.re * a.re + a.im * a.im; }
template <class T> T arg(const complex<T>& a) { return std::atan2(a.im, a.re); }
template <class T> complex<T> conj(const complex<T>& a) { return complex<T>(a.re, -a.im); }
template <class T> complex<T> polar(const T& m, const T& theta = T()) { return complex<T>(m * std::cos(theta), m * std::sin(theta)); }

template <class T> complex<T> exp(const complex<T>& a) {
    const T e = std::exp(a.re);
    return complex<T>(e * std::cos(a.im), e * std::sin(a.im));
}
template <class T> complex<T> log(const complex<T>& a) { return complex<T>(std::log(abs(a)), arg(a)); }
template <class T> complex<T> log10(const complex<T>& a) { return log(a) / static_cast<T>(std::log(T(10))); }
template <class T> complex<T> sqrt(const complex<T>& a) {
    const T m = std::sqrt(abs(a));
    const T t = arg(a) * T(0.5);
    return complex<T>(m * std::cos(t), m * std::sin(t));
}
template <class T> complex<T> sin(const complex<T>& a) { return complex<T>(std::sin(a.re) * std::cosh(a.im), std::cos(a.re) * std::sinh(a.im)); }
template <class T> complex<T> cos(const complex<T>& a) { return complex<T>(std::cos(a.re) * std::cosh(a.im), -std::sin(a.re) * std::sinh(a.im)); }
template <class T> complex<T> tan(const complex<T>& a) { return sin(a) / cos(a); }
template <class T> complex<T> sinh(const complex<T>& a) { return complex<T>(std::sinh(a.re) * std::cos(a.im), std::cosh(a.re) * std::sin(a.im)); }
template <class T> complex<T> cosh(const complex<T>& a) { return complex<T>(std::cosh(a.re) * std::cos(a.im), std::sinh(a.re) * std::sin(a.im)); }
template <class T> complex<T> tanh(const complex<T>& a) { return sinh(a) / cosh(a); }
template <class T> complex<T> pow(const complex<T>& a, const complex<T>& b) { return exp(b * log(a)); }
template <class T, class S> complex<T> pow(const complex<T>& a, S s) { return exp(complex<T>(static_cast<T>(s)) * log(a)); }

} // namespace thrust

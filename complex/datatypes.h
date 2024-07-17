#pragma once

#include <cstdlib>
#include <random>
#include <type_traits>
#include <string>

template<class T>
struct __align__(2*sizeof(T)) complex {
    using T_base = T;
    T real, imag;
    __host__ __device__ inline
    complex() {
        real = 0;
        imag = 0;
    }
    __host__ __device__ inline
    complex(int real):
        real(real), imag(0) {}
    __host__ __device__ inline
    complex(T real, T imag):
        real(real), imag(imag) {}
    __host__ __device__ inline
    complex(const complex<T> & other):
        real(other.real), imag(other.imag) {}
    __host__ __device__ inline
    complex<T> & operator = (const complex<T> & other) {
        real = other.real;
        imag = other.imag;
        return *this;
    }
    __host__ __device__ inline
    void operator += (const complex<T> & right) {
        real += right.real;
        imag += right.imag;
    }
    __host__ __device__ inline
    void mac(const complex<T> & left, const complex<T> & right) {
        real += left.real*right.real - left.imag*right.imag;
        imag += left.imag*right.real + left.real*right.imag;
    }
    T norm2() const {
        return real*real + imag*imag;
    }
};

template<class T>
__host__ __device__ inline
complex<T> operator * (const complex<T> & left, const complex<T> & right) {
    return complex<T>(
        left.real*right.real - left.imag*right.imag,
        left.imag*right.real + left.real*right.imag
    );
}
template<class T>
__host__ __device__ inline
complex<T> operator + (const complex<T> & left, const complex<T> & right) {
    return complex<T>(
        left.real + right.real,
        left.imag + right.imag
    );
}
template<class T>
__host__ __device__ inline
complex<T> operator - (const complex<T> & left, const complex<T> & right) {
    return complex<T>(
        left.real - right.real,
        left.imag - right.imag
    );
}

using complexF = complex<float>;
using complexD = complex<double>;

template<class C> struct isComplex : std::false_type {};
template<class T> struct isComplex<complex<T>> : std::true_type {};

using realF = float;
using realD = double;

template<class T> __device__ inline void multiply_accumulate(T & accu, const T & a, const T & b);
template<> __device__ inline void multiply_accumulate(realF &accu, const realF &a, const realF &b) {
    accu += a * b; 
}
template<> __device__ inline void multiply_accumulate(complexF & accu, const complexF & a, const complexF & b) {
    complexF::T_base temp = accu.real - a.imag*b.imag;
    accu.real = __fmaf_rn(a.real, b.real, temp); 
    accu.imag = __fmaf_rn(a.real, b.imag, __fmaf_rn(a.imag, b.real, accu.imag)); 
}

template<class T>
bool isEqual(const T & a, const T & b);
template<> bool isEqual(const realF & a, const realF & b) { return std::abs(a-b) < 0.01; }
template<> bool isEqual(const realD & a, const realD & b) { return std::abs(a-b) < 0.01; }
template<> bool isEqual(const complexF & a, const complexF & b) { return (a-b).norm2() < 0.01; }
template<> bool isEqual(const complexD & a, const complexD & b) { return (a-b).norm2() < 0.01; }

template<class T> struct uniform_distribution;
template<> struct uniform_distribution<realF> {
    std::uniform_real_distribution<realF> dist;
    uniform_distribution(realF min, realF max):
        dist(std::uniform_real_distribution<realF>(min, max))
    {}
    template<class Gen>
    realF operator () (Gen & gen) {
        return dist(gen);
    }
};
template<> struct uniform_distribution<realD> {
    std::uniform_real_distribution<realD> dist;
    uniform_distribution(realD min, realD max):
        dist(std::uniform_real_distribution<realD>(min, max))
    {}
    template<class Gen>
    realD operator () (Gen & gen) {
        return dist(gen);
    }
};
template<class T> struct uniform_distribution<complex<T>> {
    std::uniform_real_distribution<T> dist_real;
    std::uniform_real_distribution<T> dist_imag;
    uniform_distribution(const complex<T> & min, const complex<T> & max):
        dist_real(std::uniform_real_distribution<T>(min.real, max.real)),
        dist_imag(std::uniform_real_distribution<T>(min.imag, max.imag))
    {}
    template<class Gen>
    complex<T> operator () (Gen & gen) {
        return complex<T>(dist_real(gen), dist_imag(gen));
    }
};

template<class T> struct typeAsString;
template<> struct typeAsString<realF> { static constexpr char value[] = "realF"; };
template<> struct typeAsString<realD> { static constexpr char value[] = "realD"; };
template<> struct typeAsString<complexF> { static constexpr char value[] = "complexF"; };
template<> struct typeAsString<complexD> { static constexpr char value[] = "complexD"; };

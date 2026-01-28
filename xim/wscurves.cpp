/*
 * Copyright (c) 2026, Attila Magyar
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cmath>


#if defined(__GNUC__) || defined(__clang__)
  #define XIM_LIKELY(condition) __builtin_expect((condition), 1)
  #define XIM_UNLIKELY(condition) __builtin_expect((condition), 0)
#else
  #define XIM_LIKELY(condition) (condition)
  #define XIM_UNLIKELY(condition) (condition)
#endif


template<typename NumberType>
constexpr NumberType _db_to_linear(NumberType const db)
{
    return pow(10.0, db / 20.0);
}


template<typename NumberType>
constexpr NumberType _linear_to_db(NumberType const linear)
{
    NumberType const lin_abs = abs(linear);

    if (lin_abs < 1e-6) {
        return -120.0;
    }

    return log10(lin_abs) * 20.0;
}


constexpr size_t WS_CURVE_SIZE = 2048;
constexpr size_t WS_CURVE_MAX_IDX = WS_CURVE_SIZE - 1;

constexpr double WS_LIMIT_DB = 15.0;
constexpr double WS_MIN_LINEAR = -_db_to_linear(WS_LIMIT_DB);
constexpr double WS_MAX_LINEAR = _db_to_linear(WS_LIMIT_DB);
constexpr double WS_RANGE_LINEAR = WS_MAX_LINEAR - WS_MIN_LINEAR;
constexpr double WS_LINEAR_TO_CURVE_INDEX_SCALE = (
    static_cast<double>(WS_CURVE_MAX_IDX) / WS_RANGE_LINEAR
);


template<typename NumberType>
constexpr NumberType _ws_curve_lookup(NumberType const* const curve, NumberType const x)
{
    constexpr NumberType shift = static_cast<NumberType>(WS_MIN_LINEAR);
    constexpr NumberType scale = static_cast<NumberType>(WS_LINEAR_TO_CURVE_INDEX_SCALE);

    NumberType const index = (x - shift) * scale;

    if (XIM_UNLIKELY(index <= 0.0)) {
        return curve[0];
    }

    NumberType const before_index = std::floor(index);
    size_t const int_before_index = static_cast<size_t>(before_index);

    if (XIM_UNLIKELY(int_before_index >= WS_CURVE_MAX_IDX)) {
        return curve[WS_CURVE_MAX_IDX];
    }

    NumberType const after_weight = index - before_index;
    size_t const int_after_index = int_before_index + 1;
    NumberType const before = curve[int_before_index];
    NumberType const after = curve[int_after_index];

    return after_weight * (after - before) + before;
}


template<typename NumberType>
constexpr NumberType _ws_ad_curve_lookup(NumberType const* const ad_curve, NumberType const x)
{
    if (XIM_UNLIKELY(x <= WS_MIN_LINEAR || x >= WS_MAX_LINEAR)) {
        return x;
    }

    return _ws_curve_lookup(ad_curve, x);
}


template<typename NumberType>
constexpr std::array<NumberType, WS_CURVE_SIZE> compute_curve(NumberType(*f)(NumberType const x))
{
    std::array<NumberType, WS_CURVE_SIZE> curve{};

    for (size_t i = 0; i != WS_CURVE_SIZE; ++i) {
        double const l = static_cast<double>(i) / static_cast<double>(WS_CURVE_MAX_IDX);
        double const x = WS_RANGE_LINEAR * l + WS_MIN_LINEAR;

        curve[i] = static_cast<NumberType>(f(x));
    }

    return curve;
}


class ClipperCurveCalculator
{
    public:
        template<typename NumberType>
        static constexpr std::array<NumberType, WS_CURVE_SIZE> compute_clipper_ad_curve()
        {
            return compute_curve<NumberType>(
                [](NumberType const x) -> NumberType {
                    return f_ad<NumberType>(x);
                }
            );
        }

        template<typename NumberType>
        static constexpr std::array<NumberType, WS_CURVE_SIZE> compute_clipper_curve()
        {
            return compute_curve<NumberType>(
                [](NumberType const x) -> NumberType {
                    return f<NumberType>(x);
                }
            );
        }

    private:
        /* Constants calculated in _clipper.py */
        static constexpr double T = 0.7943282347242815;
        static constexpr double U = 1.4125375446227544;
        static constexpr double V = 0.9440608762859234;
        static constexpr double W = 5.623413251903491;

        static constexpr double a = 1.4254024461041104;
        static constexpr double b = -5.503701576221022;
        static constexpr double c = 7.045386688349519;
        static constexpr double d = -2.0438139138069396;
        static constexpr double e = -0.0031547926028702156;
        static constexpr double g = 0.03548140505997477;
        static constexpr double h = 0.9002366982947906;
        static constexpr double m = 0.4938644548598521;
        static constexpr double n = -0.42030936543965014;

        /*
        Polynomials calculated using Horner's method:
        https://en.wikipedia.org/wiki/Horner%27s_method
        */

        template<typename NumberType>
        static constexpr NumberType p(NumberType const x)
        {
            return x;
        }

        template<typename NumberType>
        static constexpr NumberType p_ad(NumberType const x)
        {
            return x * x / 2.0;
        }

        template<typename NumberType>
        static constexpr NumberType q(NumberType const x)
        {
            return ((a * x + b) * x + c) * x + d;
        }

        template<typename NumberType>
        static constexpr NumberType q_ad(NumberType const x)
        {
            return ((((a / 4.0) * x + b / 3.0) * x + c / 2.0) * x + d) * x + m;
        }

        template<typename NumberType>
        static constexpr NumberType r(NumberType const x)
        {
            return (e * x + g) * x + h;
        }

        template<typename NumberType>
        static constexpr NumberType r_ad(NumberType const x)
        {
            return (((x / 3.0) * x + g / 2.0) * x + h) * x + n;
        }

        template<typename NumberType>
        static constexpr NumberType f(NumberType const x)
        {
            if (x < -W) {
                return static_cast<NumberType>(-1.0);
            }

            if (-W <= x && x < -U) {
                return -r<NumberType>(-x);
            }

            if (-U <= x && x < -T) {
                return -q<NumberType>(-x);
            }

            if (-T <= x && x < T) {
                return p<NumberType>(x);
            }

            if (T <= x && x < U) {
                return q<NumberType>(x);
            }

            if (U <= x && x < W) {
                return r<NumberType>(x);
            }

            return static_cast<NumberType>(1.0);
        }

        template<typename NumberType>
        static constexpr NumberType f_ad(NumberType const x)
        {
            if (x < -W) {
                return -x;
            }

            if (-W <= x && x < -U) {
                return r_ad<NumberType>(-x);
            }

            if (-U <= x && x < -T) {
                return q_ad<NumberType>(-x);
            }

            if (-T <= x && x < T) {
                return p_ad<NumberType>(x);
            }

            if (T <= x && x < U) {
                return q_ad<NumberType>(x);
            }

            if (U <= x && x < W) {
                return r_ad<NumberType>(x);
            }

            return x;
        }
};


constexpr std::array<double, WS_CURVE_SIZE> CLIPPER_AD_TABLE_D = (
    ClipperCurveCalculator::compute_clipper_ad_curve<double>()
);
constexpr std::array<double, WS_CURVE_SIZE> CLIPPER_TABLE_D = (
    ClipperCurveCalculator::compute_clipper_curve<double>()
);

constexpr std::array<float, WS_CURVE_SIZE> CLIPPER_AD_TABLE_F = (
    ClipperCurveCalculator::compute_clipper_ad_curve<float>()
);
constexpr std::array<float, WS_CURVE_SIZE> CLIPPER_TABLE_F = (
    ClipperCurveCalculator::compute_clipper_curve<float>()
);

constexpr double const* P_CLIPPER_AD_TABLE_D = CLIPPER_AD_TABLE_D.data();
constexpr double const* P_CLIPPER_TABLE_D = CLIPPER_TABLE_D.data();

constexpr float const* P_CLIPPER_AD_TABLE_F = CLIPPER_AD_TABLE_F.data();
constexpr float const* P_CLIPPER_TABLE_F = CLIPPER_TABLE_F.data();


template<typename NumberType>
constexpr std::array<NumberType, WS_CURVE_SIZE> compute_tanh_ad_curve()
{
    return compute_curve<NumberType>(
        [](NumberType const x) -> NumberType {
            constexpr NumberType c = WS_MAX_LINEAR - log(cosh(WS_MAX_LINEAR));

            return log(cosh(x)) + c;
        }
    );
}


template<typename NumberType>
constexpr std::array<NumberType, WS_CURVE_SIZE> compute_tanh_curve()
{
    return compute_curve<NumberType>(
        [](NumberType const x) -> NumberType {
            return tanh(x);
        }
    );
}


constexpr std::array<double, WS_CURVE_SIZE> TANH_AD_TABLE_D = compute_tanh_ad_curve<double>();
constexpr std::array<double, WS_CURVE_SIZE> TANH_TABLE_D = compute_tanh_curve<double>();

constexpr std::array<float, WS_CURVE_SIZE> TANH_AD_TABLE_F = compute_tanh_ad_curve<float>();
constexpr std::array<float, WS_CURVE_SIZE> TANH_TABLE_F = compute_tanh_curve<float>();

constexpr double const* P_TANH_AD_TABLE_D = TANH_AD_TABLE_D.data();
constexpr double const* P_TANH_TABLE_D = TANH_TABLE_D.data();

constexpr float const* P_TANH_AD_TABLE_F = TANH_AD_TABLE_F.data();
constexpr float const* P_TANH_TABLE_F = TANH_TABLE_F.data();

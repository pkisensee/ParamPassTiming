//  Parameter passing evaluation for C++
// 
//  Copyright © Pete Isensee (PKIsensee@msn.com).
//  All rights reserved worldwide.
//
//  Permission to copy, modify, reproduce or redistribute this source code is
//  granted provided the above copyright notice is retained in the resulting 
//  source code.
// 
//  This software is provided "as is" and without any express or implied
//  warranties.
// 
//  MSVC: /std:c++20 /O2
//  GCC, clang: -std=c++20 -O2

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdint>
#include <format>
#include <iostream>
#include <ranges>

#pragma warning( disable: 4068 ) // Visual Studio: "unknown pragma" warning

// Flag key functions as non-inline so that params are actually passed
#if defined(_MSC_VER)
  #define NOINLINE __declspec( noinline )
  #define CDECL __cdecl
#else
  #define NOINLINE __attribute__( ( noinline ) )
  #define CDECL 
#endif

using sysclk = std::chrono::high_resolution_clock;
using units = std::chrono::milliseconds; // typically milliseconds or microseconds

// Must be power of 2; evaluates up through kMaxArraySizex2/2.
// Typical on modern processors to see passing by value costs kick in around array size 256
static constexpr size_t kMaxArraySizex2 = 4096;

// Evaluation code does some side work to avoid potential compiler optimizations.
// These values control the amount of extra work done; keep them small
static constexpr size_t kValuesToSumMax = 4;
static constexpr size_t kSlots = 16;
std::array<size_t, kSlots> gWork;
std::vector<size_t> arrByValSums;
std::vector<size_t> arrByRefSums;

// Tailor per platform and compiler.
// Rule of thumb for analysis: total time = minutes to hours.
// Increasing kTotalRuns reduces the variance in results. The median run is selected.
// kTotalRuns must be odd (reduces extra code in Median function)
// kTotalPasses increases the number of tests performed.
static constexpr size_t kTotalRuns =
#if defined(_DEBUG)
  3;
#else
  31;
#endif

static constexpr size_t kTotalPasses =
#if defined(_DEBUG)
  100'000;
#else
  100'000'000;
#endif

// Conversion between array indices and sizeof(T)
constexpr size_t Exp2( size_t exponent )
{
  assert( exponent < ( sizeof( size_t ) * CHAR_BIT ) );
  return size_t(1) << exponent;
}

constexpr size_t Log2( size_t powerOf2 )
{
  assert( std::has_single_bit(powerOf2) );
  size_t result = 0u;
  while( powerOf2 >>= 1 )
    ++result;
  return result;
}

// Evaluate integer types through and including 64-bit integers
static constexpr size_t kCountOfElemSizesToEval = Log2( sizeof( uint64_t ) ) + 1;

// Evaluate array sizes up to and including kMaxArraySizex2/2
static constexpr size_t kCountOfArraySizesToEval = Log2( kMaxArraySizex2 );

template <typename Container>
auto GetMedian( Container& c )
{
  assert( ( std::size( c ) ) % 2 == 1 ); // simplifies the calculation
  auto half = std::size( c ) / 2;
  auto mid = std::begin( c ) + static_cast<ptrdiff_t>( half );
  std::ranges::nth_element( c, mid );
  return *mid;
}

class TimingData
{
public:
  template <typename T>
  void SetTimes( size_t arraySize, size_t run, units baseline, units byRef, units byVal )
  {
    constexpr auto elemSizeIdx = Log2( sizeof( T ) ); // element size = sizeof(T)
    const auto arrSizeIdx = Log2( arraySize );
    baselines_[elemSizeIdx][arrSizeIdx][run] = baseline;
    byRefs_[elemSizeIdx][arrSizeIdx][run] = byRef;
    byVals_[elemSizeIdx][arrSizeIdx][run] = byVal;
  }

  void OutputResults()
  {
    Output( byRefs_, "by ref" );
    Output( byVals_, "by val" );
  }

private:
  template <typename Container>
  void Output( Container& c, std::string_view hdr )
  {
    // Container c is either the "by ref" data or "by val" data
    for( size_t elemSizeIdx = 0; elemSizeIdx < kCountOfElemSizesToEval; ++elemSizeIdx )
    {
      const auto elemSize = Exp2( elemSizeIdx );
      for( size_t arrSizeIdx = 0; arrSizeIdx < kCountOfArraySizesToEval; ++arrSizeIdx )
      {
        units baseline = GetMedian( baselines_[elemSizeIdx][arrSizeIdx] );
        units time = GetMedian( c[elemSizeIdx][arrSizeIdx] );
        const auto arrSize = Exp2( arrSizeIdx );
        std::cout << hdr << ", " << arrSize << ", " << elemSize << ", " << time - baseline << '\n';
      }
    }
  }

private:
  using TimePerRun = std::array<units, kTotalRuns>;
  using TimePerArraySize = std::array<TimePerRun, kCountOfArraySizesToEval>;
  std::array<TimePerArraySize, kCountOfElemSizesToEval> baselines_;
  std::array<TimePerArraySize, kCountOfElemSizesToEval> byRefs_;
  std::array<TimePerArraySize, kCountOfElemSizesToEval> byVals_;

  // Example usage: byRefs_[elementSizeIndex][arraySizeIndex][runIndex] = units
};

class Timer
{
public:
  Timer() :
    start_( sysclk::now() )
  {
  }
  auto GetElapsedMs() const
  {
    auto end = sysclk::now();
    auto duration = end - start_;
    return std::chrono::duration_cast<units>( duration );
  }
private:
  sysclk::time_point start_;
};

class FauxRand
{
public:
  explicit FauxRand( size_t seed = 0 ) :
    next_( seed )
  {
  }

  void Seed( size_t seed )
  {
    next_ = seed;
  }

  auto operator()()
  {
    // Fast PRNG -- but not very random, and definitely not secure.
    // Useful for timing evaluations, but not for production
    next_ = next_ * 214013 + 2531011;
    return next_;
  }

private:
  size_t next_ = 0;
};

FauxRand rnd( /*seed:*/ 42 ); // global random generator

void RandomWork()
{
  // Do some calculuations and store the results
  auto slot = rnd() % kSlots;
  auto random = rnd();
  gWork[slot] += random ^ ( random << 1 );
}

template< typename T, size_t N >
class RandArray
{
public:
  RandArray()
  {
    // Fill with random values
    std::ranges::transform( arr_, std::begin(arr_), []( T )
      {
        return static_cast<T>( rnd() );
      } );
  }

  constexpr size_t ComputeValue() const
  {
    // Sum up a few random values from the array
    const auto numValuesToSum = rnd() % kValuesToSumMax;
    size_t sum = 0u;
    for( size_t i = 0; i < numValuesToSum; ++i )
    {
      auto slot = rnd() % N;
      sum += static_cast<size_t>( arr_[slot] );
    }
    return sum;
  }

private:
  std::array<T, N> arr_;
};

//----------------------------------------------------------------------------

// Functions to be timed; not inlined or optimized so parameters are actually passed.
// RandomWork() added to avoid any other clever compiler inlining
#pragma optimize( "", off )
#pragma GCC push_options
#pragma GCC optimize ("O0")
#pragma clang optimize off

template< typename T, size_t N >
NOINLINE auto ByVal( RandArray<T, N> arr )
{
  // arr passed by value and all elements copied
  RandomWork();
  return arr.ComputeValue();
}

template< typename T, size_t N >
NOINLINE auto ByRef( const RandArray<T, N>& arr )
{
  // arr passed by reference and not copied but dereferenced
  RandomWork();
  return arr.ComputeValue();
}

// Restore optimizations
#pragma optimize( "", on )
#pragma GCC pop_options
#pragma clang optimize on

//----------------------------------------------------------------------------

template< typename T, size_t N >
auto TimeArrBaseline( const RandArray<T, N>& arr )
{
  size_t sum = 0; // suppress compiler optimizations by capturing result
  Timer timer;
  for( size_t i = 0; i < kTotalPasses; ++i )
  {
    // Do the ByRef/ByVal work directly inline; no arrays copied
    RandomWork();
    sum += arr.ComputeValue();
  }
  auto end = timer.GetElapsedMs();
  return end;
}

template< typename T, size_t N >
auto TimeArrByVal( const RandArray<T, N>& arr )
{
  size_t sum = 0; // suppress compiler optimizations by capturing result
  Timer timer;
  for( size_t i = 0; i < kTotalPasses; ++i )
    sum += ByVal( arr ); // arr copied
  auto end = timer.GetElapsedMs();
  arrByValSums.push_back( sum );
  return end;
}

template< typename T, size_t N >
auto TimeArrByRef( const RandArray<T, N>& arr )
{
  size_t sum = 0; // suppress compiler optimizations by capturing result
  Timer timer;
  for( size_t i = 0; i < kTotalPasses; ++i )
    sum += ByRef( arr ); // arr passed by reference
  auto end = timer.GetElapsedMs();
  arrByRefSums.push_back( sum );
  return end;
}

TimingData timingData; // global timing data

// Generate differently sized RandArray objects at runtime in powers of 2
template< typename T, size_t MaxArrSize, size_t ArrSize = 1 >
struct EvalParamPassing
{
  static void Eval(size_t run)
  {
    RandArray<T, ArrSize> arr;
    static_assert( sizeof( arr ) == ArrSize * sizeof(T) );

    std::cout << "timing arr size=" << ArrSize << " sizeof(T)=" << sizeof( T ) << '\n';
    auto baseline = TimeArrBaseline( arr );
    auto byRef = TimeArrByRef( arr );
    auto byVal = TimeArrByVal( arr );
    timingData.SetTimes<T>( ArrSize, run, baseline, byRef, byVal );

    // Recursively evaluate the next size
    EvalParamPassing<T, MaxArrSize, ArrSize * 2>::Eval(run);
  }
};

// Base case
template< typename T, size_t MaxArrSize >
struct EvalParamPassing<T, MaxArrSize, MaxArrSize>
{
  static void Eval(size_t) {}  // recursion ends at MaxArrSize
};

int CDECL main()
{
  // simplifies median calculations
  static_assert( (kTotalRuns % 2) == 1, "Runs must be an odd number" );

  std::cout << "Running timing evaluation\n";
  for( size_t run = 0; run < kTotalRuns; ++run )
  {
    std::cout << "Run # " << run << '\n';
    EvalParamPassing<uint8_t, kMaxArraySizex2>::Eval(run);
    EvalParamPassing<uint16_t, kMaxArraySizex2>::Eval(run);
    EvalParamPassing<uint32_t, kMaxArraySizex2>::Eval(run);
    EvalParamPassing<uint64_t, kMaxArraySizex2>::Eval(run);
  }

  // Display timing results
  std::cout << "\nParam Pass Type, Array Size, sizeof(T), Time" << '\n';
  timingData.OutputResults();

  // Computed results suppress compiler optimizations
  return static_cast<int>( arrByValSums.front() + arrByRefSums.front() );
}

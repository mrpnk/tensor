#pragma once
#define FMT_HEADER_ONLY
#include "fmt/format.h"
#include "fmt/color.h"
#include <fmt/ostream.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <numeric>
#include <iomanip>
#include <functional>
#include <memory>
#include <algorithm>
#include <omp.h>
#include <utility>

#if defined(__linux__ )
inline std::string _normal_func_name(std::string full) {
	int i = full.find("::");
	i = full.rfind(" ", i);
	full = full.substr(i + 1);
	i = full.find("(");
	return full.substr(0, i);
}
#define _FUNC_ _normal_func_name(__PRETTY_FUNCTION__)
#else
#define _FUNC_ __FUNCTION__
#endif


#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif


class Timer
{
	std::chrono::high_resolution_clock hrc;
	using duration_t = std::chrono::duration<long long, std::nano>;
	struct Entry
	{
		std::string name, fullName;
		int count;
		float time;
		Entry* mommy = nullptr;
		std::vector<Entry*> children;
		std::chrono::time_point<decltype(hrc)> startTime;
	};
	std::map<std::string, Entry*> entries;
	Entry* current = nullptr;

	auto now() const {
		return std::chrono::high_resolution_clock::now();
		/*std::chrono::steady_clock::duration d(__rdtsc());
		return std::chrono::steady_clock::time_point(d);*/
	}
public:
	Timer()
	{
		current = new Entry{ "" };
	}
	~Timer()
	{
		std::function<void(Entry*)> cleanup = [&](Entry* e) {
			for (auto& a : e->children) { cleanup(a); a = nullptr; }
		};
		cleanup(current);
		delete current;
		current = nullptr;
	}
	void start(std::string const& cat)
	{
		auto fullName = current->fullName + "/" + cat;
		Entry* e = entries[fullName];
		if (!e)
		{
			e = new Entry{ cat,fullName,0,0,current };
			entries[fullName] = e;
			current->children.push_back(e);
		}
		e->startTime = now();
		current = e;
	}
	float end()
	{
		using namespace std::chrono;
		if (!current->mommy)
		{
			fmt::print(fg(fmt::color::orange), "WARNING: Timer stopped more often than started.\n");
			return -1;
		}
		auto end = now();
		auto passedSeconds = 1e-9f * duration_cast<nanoseconds>(end - current->startTime).count();
		current->count++;
		current->time += passedSeconds;
		current = current->mommy;
		return passedSeconds;
	}
	duration_t getCurrentDuration() const {
		return now() - current->startTime;
	}
	float getTotalSeconds(std::string const& name) const {
		if (!entries.contains("/"+name))
			return 0;
		return entries.at("/"+name)->time;
	}
	const auto& getEntries() const {
		return entries;
	}
	void print(std::ostream& os = std::cout, bool formatOutput = true) const
	{
		using namespace std;

		fmt::print(fmt::text_style(), "\n{}\n", string(83, '='));

		fmt::text_style rowCols[] = { fmt::text_style(), bg(fmt::color::dark_slate_gray)};
		int rowIdx = 0;
		fmt::print(bg(fmt::color::teal),
				   "{:<46} : {:>8} | {:>10} | {:>10}","Function","Count","Time [s]","Time/Call");
		fmt::print(rowCols[0], "\n");
		std::function<void(Entry*, int, bool)> printEntry = [&](Entry* e, int level, bool lastChild)
		{
			if (!e->fullName.empty()) {
				std::string ph = ""; for (int i = 0; i < std::max(0, level - 1); ++i) ph += "┆ ";
				fmt::print(rowCols[(rowIdx++) % 2], "{:<46} : {:>8} | {:>10.6f} | {:>10.6f}",
				           ph + (level ? lastChild ? "└╴" : "├╴" : "") + e->name,
				           e->count, e->time, (e->time / e->count));

				fmt::print(rowCols[0], "\n");
			}
			for (int i = 0;i<e->children.size();++i)
				printEntry(e->children[i], level + 1, i == e->children.size()-1);
		};
		Entry* root = current; while (root->mommy) { root = root->mommy; }
		printEntry(root, -1, false);

		fmt::print(fmt::text_style(), "\n{}\n", string(83, '='));
	}
};
enum verbosity {
	eNothing = 0,   // error
	eBasic = 1,     // warning
	eAdditional = 2 // info
};
#if defined DEBUG
	verbosity g_verb = eAdditional;
#else
	verbosity g_verbosity = eAdditional;
#endif

class AutoTimer
{
	Timer* timer = nullptr;
public:

	// Logs time if v <= g_verb
	AutoTimer(Timer& t, std::string const& cat, verbosity v = eBasic)
	{
		if (v <= g_verbosity) {
			if (omp_in_parallel())
				return; // Timer is not thread safe. Avoid UB.
			timer = std::addressof(t);
			timer->start(cat);
		}
	}
	~AutoTimer()
	{
		if (timer)
			timer->end();
		timer = nullptr;
	}
};


inline Timer g_timer;
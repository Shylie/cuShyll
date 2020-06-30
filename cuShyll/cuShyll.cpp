#include "pch.h"
#include "Tokenizer.h"
#include "Repr.h"
#include "Parser.h"
#include "Converter.h"

#include <thread>
#include <future>
#include <chrono>

int main(int argc, char** argv)
{
	switch (argc)
	{
	case 3:
	{
		std::ifstream in;
		in.open(argv[1]);
		if (!in.is_open()) { return EXIT_FAILURE; }

		std::string str(static_cast<const std::stringstream&>(std::stringstream() << in.rdbuf()).str());

		Converter converter(str);

		auto output = std::async(std::launch::async, &Converter::operator(), &converter);

		auto res = output.wait_for(std::chrono::seconds(5));
		if (res == std::future_status::ready)
		{
			std::ofstream file;
			file.open(argv[2]);
			if (!file.is_open()) { return EXIT_FAILURE; }
			file << output.get();
			file.close();
			return EXIT_SUCCESS;
		}
		else
		{
			return EXIT_FAILURE;
		}
	}

	case 4:
	{
		std::ifstream in;
		in.open(argv[1]);
		if (!in.is_open()) { return EXIT_FAILURE; }

		std::string str(static_cast<const std::stringstream&>(std::stringstream() << in.rdbuf()).str());

		Converter converter(str, argv[3]);

		auto output = std::async(std::launch::async, &Converter::operator(), &converter);

		auto res = output.wait_for(std::chrono::seconds(5));
		if (res == std::future_status::ready)
		{
			std::ofstream file;
			file.open(argv[2]);
			if (!file.is_open()) { return EXIT_FAILURE; }
			file << output.get();
			file.close();
			return EXIT_SUCCESS;
		}
		else
		{
			return EXIT_FAILURE;
		}
	}

	default:
		std::cerr << "Usage: cuShyll source output [method prefix]";
		return EXIT_FAILURE;
	}
}
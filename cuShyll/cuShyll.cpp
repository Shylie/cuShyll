#include "pch.h"
#include "Tokenizer.h"
#include "Repr.h"
#include "Parser.h"
#include "Converter.h"

#include <thread>
#include <future>
#include <chrono>

/*
baseclass Material
- bool DoSomething(int arg)
- bool DoSomethingElse(int arg2)
= int data

subclass Lambertian implements Material
- bool DoSomething(int arg)
{
	return arg > data;
}
- bool DoSomethingElse(int arg2)
{
	return arg2 < data + data2;
}
= int data2

subclass Metal implements Material
- bool DoSomething(int arg)
{
	return arg + data > data2 + data3;
}
- bool DoSomethingElse(int arg2)
{
	return arg2 - data < data3 - data2;
}
= int data2
= int data3
*/

/*
union MaterialData
{
	struct
	{
		int data;
		int data2;
	} lambertian;
	struct
	{
		int data;
		int data2;
		int data3;
	} metal;
};

bool LambertianDoSomething(int arg, MaterialData data);
bool LambertianDoSomethingElse(int arg2, MaterialData data);
bool MetalDoSomething(int arg, MaterialData data);
bool MetalDoSomethingElse(int arg2, MaterialData data);

struct Material
{
	enum class Type
	{
		Lambertian,
		Metal
	} type;

	MaterialData data;

	bool DoSomething(int arg)
	{
		switch (type)
		{
		case Type::Lambertian:
			return LambertianDoSomething(arg, data);

		case Type::Metal:
			return MetalDoSomething(arg, data);

		default:
			return false; // some default behavior. not sure how to deal with this yet
		}
	}

	bool DoSomethingElse(int arg2)
	{
		switch (type)
		{
		case Type::Lambertian:
			return LambertianDoSomethingElse(arg2, data);
		
		case Type::Metal:
			return MetalDoSomethingElse(arg2, data);

		default:
			return false; // some default behavior. not sure how to deal with this yet
		}
	}
};

bool LambertianDoSomething(int arg, MaterialData data)
{
	return arg > data.lambertian.data;
}

bool LambertianDoSomethingElse(int arg2, MaterialData data)
{
	return arg2 < data.lambertian.data + data.lambertian.data2;
}

bool MetalDoSomething(int arg, MaterialData data)
{
	return arg + data.metal.data > data.metal.data2 + data.metal.data3;
}

bool MetalDoSomethingElse(int arg2, MaterialData data)
{
	return arg2 - data.metal.data < data.metal.data3 - data.metal.data2;
}
*/

int main(int argc, char** argv)
{
	if (argc != 3) return EXIT_FAILURE;

	std::ifstream in;
	in.open(argv[1]);
	if (!in.is_open()) return EXIT_FAILURE;

	std::string str(static_cast<std::stringstream const&>(std::stringstream() << in.rdbuf()).str());

	Converter converter(str);

	auto output = std::async(&Converter::operator(), &converter);

	auto res = output.wait_for(std::chrono::seconds(5));
	if (res == std::future_status::ready)
	{
		std::ofstream file;
		file.open(argv[2]);
		if (!file.is_open()) return EXIT_FAILURE;
		file << converter();
		file.close();
		return EXIT_SUCCESS;
	}
	else
	{
		return EXIT_FAILURE;
	}
}
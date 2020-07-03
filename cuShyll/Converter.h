#pragma once

#include "pch.h"
#include "Tokenizer.h"
#include "Repr.h"
#include "Parser.h"

class Converter final
{
public:
	Converter(const std::string& source, const char* prefix = nullptr);

	std::string operator()();

private:
	std::string ConvertBaseDataUnion(size_t base);
	std::string ConvertDataUnion(size_t base, size_t sub);
	std::string ConvertDataUnionConstructor(size_t base, size_t sub);
	std::string ConvertSubMethod(size_t base, size_t sub, bool forward);
	std::string ConvertBase(size_t base);
	std::string ConvertBaseMethod(size_t base, size_t method);
	std::string ConvertSubFactory(size_t base, size_t sub, bool forward);

	std::vector<Hierarchy> hierarchies;
	const char* prefix;
};
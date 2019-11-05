#pragma once

#include "pch.h"
#include "Tokenizer.h"
#include "Repr.h"

class Parser final
{
public:
	Parser(std::string source);

	bool Parse(std::vector<Hierarchy>& hierarchies);

private:
	void BaseClass();
	void BaseMethod(Hierarchy& hierarchy);
	void BaseData(Hierarchy& hierarchy);

	void SubClass();
	void SubMethod(SubClassRepr& subclass);
	void SubData(SubClassRepr& subclass);

	void Advance();
	bool Match(Token::Type type);
	void Consume(Token::Type type, std::string message);

	void ErrorAt(Token token, std::string message);

	size_t FindHierarchy(std::string name);
	size_t FindSubclass(const Hierarchy& hierarchy, std::string name);

	Tokenizer tokenizer;
	std::vector<Hierarchy> hierarchies;

	Token current;
	Token previous;

	bool hadError;
};
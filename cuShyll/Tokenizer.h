#pragma once

#include "pch.h"

struct Token final
{
	enum class Type
	{
		Baseclass,
		Subclass,
		Implements,
		LeftBrace,
		RightBrace,
		LeftParen,
		RightParen,
		Comma,
		Dash,
		Equals,
		Identifier,
		CodeBlock,
		Error
	};

	Token() : type(Type::Error), lexeme(), line() { }
	Token(Type type, std::string lexeme, int line) : type(type), lexeme(lexeme), line(line) { }

	std::string lexeme;
	Type type;
	int line;
};

class Tokenizer final
{
public:
	Tokenizer(std::string source);

	Token Next();
	bool IsAtEnd() const;

private:
	char Peek(int ahead = 0) const;
	void Advance();

	std::string source;
	int start;
	int current;
	int scopeDepth;
	int line;
};
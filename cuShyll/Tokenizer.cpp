#include "pch.h"
#include "Tokenizer.h"

bool IsAlpha(char c)
{
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
}

bool IsNumeric(char c)
{
	return c >= '0' && c <= '9';
}

bool IsWhitespace(char c)
{
	return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

Tokenizer::Tokenizer(std::string source) : source(source), start(0), current(0), scopeDepth(0), line(0) { }

Token Tokenizer::Next()
{
	start = current;

	if (scopeDepth > 0)
	{
		while (scopeDepth > 0 && !IsAtEnd())
		{
			switch (Peek())
			{
			case '{':
				Advance();
				scopeDepth++;
				break;

			case '}':
				scopeDepth--;
				if (scopeDepth > 0) Advance();
				break;

			default:
				Advance();
				break;
			}
		}
		return Token(Token::Type::CodeBlock, source.substr(start, current - start), line);
	}

	while (IsWhitespace(Peek()) && !IsAtEnd())
	{
		Advance();
	}

	start = current;

	switch (Peek())
	{
	case '(':
		Advance();
		return Token(Token::Type::LeftParen, "(", line);

	case ')':
		Advance();
		return Token(Token::Type::RightParen, ")", line);

	case '{':
		Advance();
		scopeDepth++;
		return Token(Token::Type::LeftBrace, "{", line);

	case '}':
		Advance();
		return Token(Token::Type::RightBrace, "}", line);

	case ',':
		Advance();
		return Token(Token::Type::Comma, ",", line);

	case '-':
		Advance();
		return Token(Token::Type::Dash, "-", line);

	case '=':
		Advance();
		return Token(Token::Type::Equals, "=", line);
	}

	if (IsAlpha(Peek()))
	{
		do
		{
			Advance();
		}
		while ((IsAlpha(Peek()) || IsNumeric(Peek())) && !IsAtEnd());
		while (Peek() == '&' || Peek() == '*') Advance();

		bool atEnd = IsAtEnd();
		if (atEnd) current++;
		std::string lexeme = source.substr(start, current - start);
		if (atEnd) current--;
		if (lexeme == "baseclass") return Token(Token::Type::Baseclass, lexeme, line);
		if (lexeme == "subclass") return Token(Token::Type::Subclass, lexeme, line);
		if (lexeme == "implements") return Token(Token::Type::Implements, lexeme, line);
		if (lexeme == "const" && Peek() == ' ' && IsAlpha(Peek(1)))
		{
			int nstart = current;
			do
			{
				Advance();
			} while ((IsAlpha(Peek()) || IsNumeric(Peek())) && !IsAtEnd());

			while (Peek() == '&' || Peek() == '*') Advance();


			atEnd = IsAtEnd();
			if (atEnd) current++;
			std::string nlexeme = source.substr(start, current - nstart);
			if (atEnd) current--;

			if (nlexeme == "baseclass") return Token(Token::Type::Error, lexeme, line);
			if (nlexeme == "subclass") return Token(Token::Type::Error, lexeme, line);
			if (nlexeme == "implements") return Token(Token::Type::Error, lexeme, line);

			if (atEnd) current++;
			lexeme = source.substr(start, current - start);
			if (atEnd) current--;
			return Token(Token::Type::Identifier, lexeme, line);
		}
		return Token(Token::Type::Identifier, lexeme, line);
	}

	return Token(Token::Type::Error, source.substr(start, current - start), line);
}

bool Tokenizer::IsAtEnd() const
{
	return current >= source.length() - 1;
}

char Tokenizer::Peek(int ahead) const
{
	if (current + ahead >= source.length()) return '\0';
	return source.at(current + ahead);
}

void Tokenizer::Advance()
{
	if (current < source.length() - 1)
	{
		current++;
		if (Peek(-1) == '\n') line++;
	}
}
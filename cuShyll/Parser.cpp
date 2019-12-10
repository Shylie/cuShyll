#include "pch.h"
#include "Parser.h"
#include "Tokenizer.h"
#include "Repr.h"

Parser::Parser(std::string source) : tokenizer(source), hadError(false)
{
	Advance();
}

bool Parser::Parse(std::vector<Hierarchy>& hierarchies)
{
	while (!tokenizer.IsAtEnd() && !hadError)
	{
		if (Match(Token::Type::Error))
		{
			ErrorAt(previous, "Unknown error.");
			return false;
		}
		if (Match(Token::Type::Baseclass))
		{
			BaseClass();
			continue;
		}
		if (Match(Token::Type::Subclass))
		{
			SubClass();
			continue;
		}
	}

	if (hadError) return false;

	if (!tokenizer.IsAtEnd()) ErrorAt(current, "Expected end of file.");

	hierarchies = this->hierarchies;
	return true;
}

void Parser::BaseClass()
{
	Consume(Token::Type::Identifier, "Expected identifier after 'baseclass'.");
	hierarchies.emplace_back();
	hierarchies.back().base.name = previous.lexeme;
	size_t back = hierarchies.size() - 1;
	while (!tokenizer.IsAtEnd() && (current.type == Token::Type::Dash || current.type == Token::Type::Equals))
	{
		if (Match(Token::Type::Dash))
		{
			BaseMethod(hierarchies.at(back));
		}
		else if (Match(Token::Type::Equals))
		{
			BaseData(hierarchies.at(back));
		}
	}
}

void Parser::BaseMethod(Hierarchy& hierarchy)
{
	hierarchy.base.methods.emplace_back();
	Consume(Token::Type::Identifier, "Expected type after '-'.");
	hierarchy.base.methods.back().rettype = previous.lexeme;
	Consume(Token::Type::Identifier, "Expected name after type.");
	hierarchy.base.methods.back().name = previous.lexeme;
	Consume(Token::Type::LeftParen, "Expected '(' after name.");

	while (!Match(Token::Type::RightParen) && !tokenizer.IsAtEnd())
	{
		hierarchy.base.methods.back().args.emplace_back();
		Consume(Token::Type::Identifier, "Expected type.");
		hierarchy.base.methods.back().args.back().type = previous.lexeme;
		Consume(Token::Type::Identifier, "Expected name after type.");
		hierarchy.base.methods.back().args.back().name = previous.lexeme;
		if (!Match(Token::Type::Comma) && current.type != Token::Type::RightParen)
		{
			ErrorAt(current, "Expected ',' inbetween parameters.");
			break;
		}
	}

	if (Match(Token::Type::LeftBrace))
	{
		Consume(Token::Type::CodeBlock, "Expected code block after '{'.");
		hierarchy.base.methods.back().contents = previous.lexeme;
		hierarchy.base.methods.back().hasContents = true;
		Consume(Token::Type::RightBrace, "Expected '}' after code block.");
	}
}

void Parser::BaseData(Hierarchy& hierarchy)
{
	hierarchy.base.data.emplace_back();
	Consume(Token::Type::Identifier, "Expected type after '='.");
	hierarchy.base.data.back().type = previous.lexeme;
	Consume(Token::Type::Identifier, "Expected name after type.");
	hierarchy.base.data.back().name = previous.lexeme;
}

void Parser::SubClass()
{
	Consume(Token::Type::Identifier, "Expected name after 'subclass'.");
	std::string scname = previous.lexeme;
	Consume(Token::Type::Implements, "Expected 'implements' after name.");
	Consume(Token::Type::Identifier, "Expected name after 'implements'.");
	size_t pos = FindHierarchy(previous.lexeme);
	if (pos > hierarchies.size()) ErrorAt(current, "Undefined baseclass '" + previous.lexeme + "'.");
	Hierarchy& hierarchy = hierarchies.at(pos);

	hierarchy.subclasses.emplace_back();
	hierarchy.subclasses.back().name = scname;
	hierarchy.subclasses.back().base = hierarchy.base;
	while (!tokenizer.IsAtEnd() && (current.type == Token::Type::Dash || current.type == Token::Type::Equals))
	{
		if (Match(Token::Type::Dash))
		{
			SubMethod(hierarchy.subclasses.back());
		}
		else if (Match(Token::Type::Equals))
		{
			SubData(hierarchy.subclasses.back());
		}
		else
		{
			ErrorAt(current, "Unexpected token.");
			break;
		}
	}
}

void Parser::SubMethod(SubClassRepr& subclass)
{
	Consume(Token::Type::Identifier, "Expected type after '-'.");
	std::string rettype = previous.lexeme;
	Consume(Token::Type::Identifier, "Expected name after type.");
	Token name = previous;
	Consume(Token::Type::LeftParen, "Expected '(' after name.");

	std::vector<VarRepr> args;
	while (!Match(Token::Type::RightParen) && !tokenizer.IsAtEnd())
	{
		args.emplace_back();
		Consume(Token::Type::Identifier, "Expected type.");
		args.back().type = previous.lexeme;
		Consume(Token::Type::Identifier, "Expected name after type.");
		args.back().name = previous.lexeme;
		if (!Match(Token::Type::Comma) && current.type != Token::Type::RightParen)
		{
			ErrorAt(current, "Expected ',' inbetween parameters.");
			break;
		}
	}

	if (!Match(Token::Type::LeftBrace))
	{
		bool doError = true;
		size_t matchidx = 0;
		for (size_t i = 0; i < subclass.base.methods.size(); i++)
		{
			if (subclass.base.methods.at(i).name != name.lexeme) continue;
			if (subclass.base.methods.at(i).rettype != rettype) continue;
			bool paramsMatch = (subclass.base.methods.at(i).args.size() == args.size());
			if (paramsMatch)
			{
				for (size_t j = 0; j < subclass.base.methods.at(i).args.size(); j++)
				{
					if (subclass.base.methods.at(i).args.at(j).type != args.at(j).type)
					{
						paramsMatch = false;
						break;
					}
				}
				if (paramsMatch)
				{
					doError = false;
					matchidx = i;
					break;
				}
			}
		}
		if (doError || !subclass.base.methods.at(matchidx).hasContents)
		{
			ErrorAt(name, "Expected method definition for virtual method '" + name.lexeme + "'.");
		}
		else
		{
			subclass.methods.emplace_back();
			subclass.methods.back().base = subclass.base.methods.at(matchidx);
			subclass.methods.back().hasContents = false;

		}
	}
	else
	{
		Consume(Token::Type::CodeBlock, "Expected code block after '{'.");
		std::string code = previous.lexeme;
		Consume(Token::Type::RightBrace, "Expected '}' after code block.");

		bool doError = true;
		size_t matchidx = 0;
		for (size_t i = 0; i < subclass.base.methods.size(); i++)
		{
			if (subclass.base.methods.at(i).name != name.lexeme) continue;
			if (subclass.base.methods.at(i).rettype != rettype) continue;
			bool paramsMatch = (subclass.base.methods.at(i).args.size() == args.size());
			if (paramsMatch)
			{
				for (size_t j = 0; j < subclass.base.methods.at(i).args.size(); j++)
				{
					if (subclass.base.methods.at(i).args.at(j).type != args.at(j).type)
					{
						paramsMatch = false;
						break;
					}
				}
				if (paramsMatch)
				{
					doError = false;
					matchidx = i;
					break;
				}
			}
		}
		if (doError)
		{
			ErrorAt(name, "Undefined baseclass method '" + name.lexeme + "'.");
		}
		else
		{
			subclass.methods.emplace_back();
			subclass.methods.back().base = subclass.base.methods.at(matchidx);
			subclass.methods.back().contents = code;
		}
	}
}

void Parser::SubData(SubClassRepr& subclass)
{
	subclass.data.emplace_back();
	Consume(Token::Type::Identifier, "Expected type after '='.");
	subclass.data.back().type = previous.lexeme;
	Consume(Token::Type::Identifier, "Expected name after type.");
	subclass.data.back().name = previous.lexeme;
}

void Parser::Advance()
{
	previous = current;
	if (tokenizer.IsAtEnd()) return;
	current = tokenizer.Next();
}

bool Parser::Match(Token::Type type)
{
	if (current.type != type) return false;
	Advance();
	return true;
}

void Parser::Consume(Token::Type type, std::string message)
{
	if (current.type != type)
	{
		ErrorAt(current.lexeme.length() > 0 ? current : previous, message);
	}
	else
	{
		Advance();
	}
}

void Parser::ErrorAt(Token token, std::string message)
{
	std::cerr << "Error on line " << (token.line + 1) << ": " << message << "\nat: " << token.lexeme << "\n";
	hadError = true;
}

size_t Parser::FindHierarchy(std::string name)
{
	for (size_t i = 0; i < hierarchies.size(); i++)
	{
		if (hierarchies.at(i).base.name == name) return i;
	}
	return -1; // not here?
}

size_t Parser::FindSubclass(const Hierarchy& hierarchy, std::string name)
{
	for (size_t i = 0; i < hierarchy.subclasses.size(); i++)
	{
		if (hierarchy.subclasses.at(i).name == name) return i;
	}
	return -1; // not here?
}
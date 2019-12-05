#pragma once

#include "pch.h"

struct VarRepr final
{
	std::string name;
	std::string type;
};

struct BaseMethodRepr final
{
	std::string name;
	std::string rettype;
	std::vector<VarRepr> args;
	std::string contents;
	bool hasContents = false;
};

struct SubMethodRepr final
{
	std::string contents;
	bool hasContents = true;
	BaseMethodRepr base;
};

struct BaseClassRepr final
{
	std::string name;
	std::vector<BaseMethodRepr> methods;
	std::vector<VarRepr> data;
};

struct SubClassRepr final
{
	std::string name;
	std::vector<SubMethodRepr> methods;
	std::vector<VarRepr> data;
	BaseClassRepr base;
};

struct Hierarchy final
{
	BaseClassRepr base;
	std::vector<SubClassRepr> subclasses;
};
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
	std::string contents;
	bool hasContents = false;
	std::vector<VarRepr> args;
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
	bool bitfield = false;
	std::vector<BaseMethodRepr> methods;
	std::vector<VarRepr> data;
};

struct SubClassRepr final
{
	std::string name;
	BaseClassRepr base;
	std::vector<SubMethodRepr> methods;
	std::vector<VarRepr> data;
};

struct Hierarchy final
{
	BaseClassRepr base;
	std::vector<SubClassRepr> subclasses;
};
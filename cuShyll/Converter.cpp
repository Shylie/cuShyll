#include "pch.h"
#include "Converter.h"
#include "Tokenizer.h"
#include "Repr.h"
#include "Parser.h"

Converter::Converter(const std::string& source, const char* prefix) : prefix(prefix)
{
	Parser(source).Parse(hierarchies);
}

std::string Converter::operator()()
{
	std::string temp = "";
	for (size_t i = 0; i < hierarchies.size(); i++)
	{
		temp += "struct " + hierarchies.at(i).base.name + ";\n";
	}
	temp += "\n";
	for (size_t i = 0; i < hierarchies.size(); i++)
	{
		temp += "union " + hierarchies.at(i).base.name + "Data\n{\n" + ConvertBaseDataUnion(i) + "\n";
		for (size_t j = 0; j < hierarchies.at(i).subclasses.size(); j++)
		{
			temp += ConvertDataUnion(i, j) + "\n";
		}
		temp += ConvertBaseDataUnionConstructor(i) + "\n";
		for (size_t j = 0; j < hierarchies.at(i).subclasses.size(); j++)
		{
			temp += ConvertDataUnionConstructor(i, j) + "\n";
		}
		temp += "};\n\n";
		for (size_t j = 0; j < hierarchies.at(i).subclasses.size(); j++)
		{
			temp += ConvertSubMethod(i, j, true) + "\n";
		}
		temp += ConvertBase(i);
		temp += "\n";
	}
	for (size_t i = 0; i < hierarchies.size(); i++)
	{
		for (size_t j = 0; j < hierarchies.at(i).subclasses.size(); j++)
		{
			temp += "\n" + ConvertSubFactory(i, j, true);
		}
		temp += "\n";
	}
	for (size_t i = 0; i < hierarchies.size(); i++)
	{
		for (size_t j = 0; j < hierarchies.at(i).subclasses.size(); j++)
		{
			temp += "\n" + ConvertSubMethod(i, j, false);
		}
	}
	for (size_t i = 0; i < hierarchies.size(); i++)
	{
		for (size_t j = 0; j < hierarchies.at(i).subclasses.size(); j++)
		{
			temp += "\n" + ConvertSubFactory(i, j, false);
		}
	}
	return temp;
}

std::string Converter::ConvertBaseDataUnion(size_t base)
{
	std::string temp = "\tstruct BaseData\n\t{";
	for (size_t i = 0; i < hierarchies.at(base).base.data.size(); i++)
	{
		temp += "\n\t\t" + hierarchies.at(base).base.data.at(i).type + " " + hierarchies.at(base).base.data.at(i).name + ";";
	}
	temp += "\n\t} Base;";
	return temp;
}

std::string Converter::ConvertDataUnion(size_t base, size_t sub)
{
	std::string temp = "\tstruct " + hierarchies.at(base).subclasses.at(sub).name + "Data\n\t{";
	for (size_t i = 0; i < hierarchies.at(base).base.data.size(); i++)
	{
		temp += "\n\t\t" + hierarchies.at(base).base.data.at(i).type + " " + hierarchies.at(base).base.data.at(i).name + ";";
	}
	for (size_t i = 0; i < hierarchies.at(base).subclasses.at(sub).data.size(); i++)
	{
		temp += "\n\t\t" + hierarchies.at(base).subclasses.at(sub).data.at(i).type + " " + hierarchies.at(base).subclasses.at(sub).data.at(i).name + ";";
	}
	temp += "\n\t} " + hierarchies.at(base).subclasses.at(sub).name + ";";
	return temp;
}

std::string Converter::ConvertBaseDataUnionConstructor(size_t base)
{
	return (prefix ? "\t" + std::string(prefix) + " " : "\t") + hierarchies.at(base).base.name + "Data() { }";
}

std::string Converter::ConvertDataUnionConstructor(size_t base, size_t sub)
{
	return (prefix ? "\t" + std::string(prefix) + " " : "\t") + hierarchies.at(base).base.name + "Data(" + hierarchies.at(base).subclasses.at(sub).name + "Data data) : " + hierarchies.at(base).subclasses.at(sub).name + "(data) { }";
}

std::string Converter::ConvertSubMethod(size_t base, size_t sub, bool forward)
{
	std::string temp = "";
	for (size_t i = 0; i < hierarchies.at(base).subclasses.at(sub).methods.size(); i++)
	{
		if (!hierarchies.at(base).subclasses.at(sub).methods.at(i).hasContents) { continue; }
		temp += (prefix ? std::string(prefix) + " " : "") + hierarchies.at(base).subclasses.at(sub).methods.at(i).base.rettype + " ";
		temp += hierarchies.at(base).subclasses.at(sub).name + hierarchies.at(base).subclasses.at(sub).methods.at(i).base.name + "(";
		for (size_t j = 0; j < hierarchies.at(base).subclasses.at(sub).methods.at(i).base.args.size(); j++)
		{
			temp += hierarchies.at(base).subclasses.at(sub).methods.at(i).base.args.at(j).type + " ";
			temp += hierarchies.at(base).subclasses.at(sub).methods.at(i).base.args.at(j).name + ", ";
		}
		temp += hierarchies.at(base).base.name + "Data& data, " + hierarchies.at(base).base.name + "& obj";
		if (forward)
		{
			temp += ");";
		}
		else
		{
			temp += ")\n{";
			std::string convertedContents = hierarchies.at(base).subclasses.at(sub).methods.at(i).contents;
			std::string unionName = "data." + hierarchies.at(base).subclasses.at(sub).name + ".$&";
			for (size_t j = 0; j < hierarchies.at(base).subclasses.at(sub).data.size(); j++)
			{
				std::regex data("\\b" + hierarchies.at(base).subclasses.at(sub).data.at(j).name + "\\b");
				convertedContents = std::regex_replace(convertedContents, data, unionName);
			}
			for (size_t j = 0; j < hierarchies.at(base).base.data.size(); j++)
			{
				std::regex data("\\b" + hierarchies.at(base).base.data.at(j).name + "\\b");
				convertedContents = std::regex_replace(convertedContents, data, unionName);
			}
			temp += convertedContents;
			temp += "}";
		}
		temp += "\n";
	}
	return temp;
}

std::string Converter::ConvertBase(size_t base)
{
	std::string temp = "";
	for (size_t i = 0; i < hierarchies.at(base).base.methods.size(); i++)
	{
		if (hierarchies.at(base).base.methods.at(i).hasContents)
		{
			temp += (prefix ? std::string(prefix) + " " : "") + hierarchies.at(base).base.methods.at(i).rettype + " ";
			temp += hierarchies.at(base).base.name + hierarchies.at(base).base.methods.at(i).name + "Base(";
			for (size_t j = 0; j < hierarchies.at(base).base.methods.at(i).args.size(); j++)
			{
				temp += hierarchies.at(base).base.methods.at(i).args.at(j).type + " ";
				temp += hierarchies.at(base).base.methods.at(i).args.at(j).name + ", ";
			}
			temp += hierarchies.at(base).base.name + "Data& data, " + hierarchies.at(base).base.name + "& obj)\n{";
			std::string convertedContents = hierarchies.at(base).base.methods.at(i).contents;
			std::string unionName = "data.BaseData.$&";
			for (size_t j = 0; j < hierarchies.at(base).base.data.size(); j++)
			{
				std::regex data("\\b" + hierarchies.at(base).base.data.at(j).name + "\\b");
				convertedContents = std::regex_replace(convertedContents, data, unionName);
			}
			temp += convertedContents;
			temp += "}\n";
		}
	}
	if (hierarchies.at(base).base.bitfield)
	{
		temp += "struct " + hierarchies.at(base).base.name + "\n{\n\tenum class Type\n\t{\n\t\tInvalid = 0,";
		for (size_t i = 0; i < hierarchies.at(base).subclasses.size(); i++)
		{
			temp += "\n\t\t" + hierarchies.at(base).subclasses.at(i).name + " = 1 << " + std::to_string(i) + ",";
		}
	}
	else
	{
		temp += "struct " + hierarchies.at(base).base.name + "\n{\n\tenum class Type\n\t{\n\t\tInvalid,";
		for (size_t i = 0; i < hierarchies.at(base).subclasses.size(); i++)
		{
			temp += "\n\t\t" + hierarchies.at(base).subclasses.at(i).name + ",";
		}
	}
	temp += "\n\t} type;\n\n\t" + hierarchies.at(base).base.name + "Data data;";
	temp += (prefix ? "\n\t" + std::string(prefix) + " " : "\n\t") + hierarchies.at(base).base.name + "() : type(Type::Invalid), data() { }";
	temp += (prefix ? "\n\t" + std::string(prefix) + " " : "\n\t") + hierarchies.at(base).base.name + "(Type type, " + hierarchies.at(base).base.name + "Data data) : type(type), data(data) { }";
	for (size_t i = 0; i < hierarchies.at(base).base.methods.size(); i++)
	{
		temp += "\n\n" + ConvertBaseMethod(base, i);
	}
	temp += "\n};";
	return temp;
}

std::string Converter::ConvertBaseMethod(size_t base, size_t method)
{
	std::string temp = (prefix ? "\t" + std::string(prefix) + " " : "\t") + hierarchies.at(base).base.methods.at(method).rettype + " " + hierarchies.at(base).base.methods.at(method).name + "(";
	for (size_t i = 0; i < hierarchies.at(base).base.methods.at(method).args.size(); i++)
	{
		temp += hierarchies.at(base).base.methods.at(method).args.at(i).type + " ";
		temp += hierarchies.at(base).base.methods.at(method).args.at(i).name;
		if (i < hierarchies.at(base).base.methods.at(method).args.size() - 1) { temp += ", "; }
	}
	temp += ")\n\t{\n\t\tswitch (type)\n\t\t{";
	for (size_t i = 0; i < hierarchies.at(base).subclasses.size(); i++)
	{
		if (!hierarchies.at(base).subclasses.at(i).methods.at(method).hasContents) { continue; }
		temp += "\n\t\tcase Type::" + hierarchies.at(base).subclasses.at(i).name + ":";
		temp += "\n\t\t\treturn " + hierarchies.at(base).subclasses.at(i).name + hierarchies.at(base).base.methods.at(method).name + "(";
		for (size_t j = 0; j < hierarchies.at(base).base.methods.at(method).args.size(); j++)
		{
			temp += hierarchies.at(base).base.methods.at(method).args.at(j).name + ", ";
		}
		temp += "data, *this);";
	}
	temp += "\n\t\tdefault:\n\t\t\treturn ";
	if (hierarchies.at(base).base.methods.at(method).hasContents)
	{
		temp += hierarchies.at(base).base.name + hierarchies.at(base).base.methods.at(method).name + "Base(";
		for (size_t j = 0; j < hierarchies.at(base).base.methods.at(method).args.size(); j++)
		{
			temp += hierarchies.at(base).base.methods.at(method).args.at(j).name + ", ";
		}
		temp += "data, *this);";
	}
	else
	{
		temp += hierarchies.at(base).base.methods.at(method).rettype + "();";
	}
	temp += "\n\t\t}\n\t}";
	return temp;
}

std::string Converter::ConvertSubFactory(size_t base, size_t sub, bool forward)
{
	std::string temp = (prefix ? std::string(prefix) + " " : "") + hierarchies.at(base).base.name + " " + hierarchies.at(base).subclasses.at(sub).name + "(";
	for (size_t i = 0; i < hierarchies.at(base).base.data.size(); i++)
	{
		temp += hierarchies.at(base).base.data.at(i).type + " ";
		temp += hierarchies.at(base).base.data.at(i).name;
		if (i < hierarchies.at(base).subclasses.at(sub).data.size() + hierarchies.at(base).base.data.size() - 1) { temp += ", "; }
	}
	for (size_t i = 0; i < hierarchies.at(base).subclasses.at(sub).data.size(); i++)
	{
		temp += hierarchies.at(base).subclasses.at(sub).data.at(i).type + " ";
		temp += hierarchies.at(base).subclasses.at(sub).data.at(i).name;
		if (i < hierarchies.at(base).subclasses.at(sub).data.size() - 1) { temp += ", "; }
	}
	if (forward)
	{
		temp += ");";
	}
	else
	{
		temp += ")\n{\n\t" + hierarchies.at(base).base.name + "Data::" + hierarchies.at(base).subclasses.at(sub).name + "Data data;";
		for (size_t i = 0; i < hierarchies.at(base).base.data.size(); i++)
		{
			temp += "\n\tdata." + hierarchies.at(base).base.data.at(i).name + " = " + hierarchies.at(base).base.data.at(i).name + ";";
		}
		for (size_t i = 0; i < hierarchies.at(base).subclasses.at(sub).data.size(); i++)
		{
			temp += "\n\tdata." + hierarchies.at(base).subclasses.at(sub).data.at(i).name + " = " + hierarchies.at(base).subclasses.at(sub).data.at(i).name + ";";
		}
		temp += "\n\treturn " + hierarchies.at(base).base.name + "(" + hierarchies.at(base).base.name + "::Type::" + hierarchies.at(base).subclasses.at(sub).name + ", data);\n}";
	}
	return temp;
}
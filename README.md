# cuShyll
cuShyll is a pseudocode to C++ transpiler to aid in writing object-oriented code for [CUDA](https://en.wikipedia.org/wiki/CUDA) programs.
## Why it exists
Virtual methods exist only on the side (CPU or GPU) that the object was created on, so passing an object made on the CPU to the GPU doesn't work, as the vtable holds incorrect addresses. The workaround to this is very convoluted and is more work than just writing code without virtual methods.
## How cuShyll solves this problem
cuShyll outputs a tagged union inside a struct to indicate the type of the object, and upon calling a "virtual method", actually calls a function based on the type held in the tag. by calling the function this way, there are no vtables.
### Example input
```
baseclass Tile
- void Tick(int i, int j, World& world)
- Color TypeColor()
= long clock

subclass Air implements Tile
- void Tick(int i, int j, World& world)
[
	clock = -1;
]
- Color TypeColor()
[
	return BLANK;
]

subclass Sand implements Tile
- void Tick(int i, int j, World& world)
[
	if (j > 0)
	{
		if (world.At(i, j - 1).clock < 0 || world.At(i, j - 1).type == Tile::Type::Fire)
		{
			world.At(i, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = Air(-1);
			return;
		}
		else if (world.At(i, j - 1).type == Tile::Type::Water)
		{
			world.At(i, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = Water(world.Clock() + 1);
			return;
		}
		else if (i > 0 && (world.At(i - 1, j).clock < 0 || world.At(i - 1, j).type == Tile::Type::Fire) && (world.At(i - 1, j - 1).clock < 0 || world.At(i - 1, j - 1).type == Tile::Type::Fire))
		{
			world.At(i - 1, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = Air(-1);
			return;
		}
		else if (i > 0 && (world.At(i - 1, j).clock < 0 || world.At(i - 1, j).type == Tile::Type::Water || world.At(i - 1, j).type == Tile::Type::Fire) && world.At(i - 1, j - 1).type == Tile::Type::Water)
		{
			world.At(i - 1, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = (world.At(i - 1, j).clock < 0 ? Air(-1) : Water(world.Clock() + 1));
			world.At(i - 1, j) = Water(world.Clock() + 1);
			return;
		}
		else if (i < world.WIDTH - 1 && (world.At(i + 1, j).clock < 0 || world.At(i + 1, j).type == Tile::Type::Fire) && (world.At(i + 1, j - 1).clock < 0 || world.At(i + 1, j - 1).type == Tile::Type::Fire))
		{
			world.At(i + 1, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = Air(-1);
			return;
		}
		else if (i < world.WIDTH - 1 && (world.At(i + 1, j).clock < 0 || world.At(i + 1, j).type == Tile::Type::Water || world.At(i + 1, j).type == Tile::Type::Fire) && world.At(i + 1, j - 1).type == Tile::Type::Water)
		{
			world.At(i + 1, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = (world.At(i + 1, j).clock < 0 ? Air(-1) : Water(world.Clock() + 1));
			world.At(i + 1, j) = Water(world.Clock() + 1);
			return;
		}
	}
	clock++;
]
- Color TypeColor()
[
	return Color{ 237, 201, 175, 255 };
]

subclass Stone implements Tile
- void Tick(int i, int j, World& world)
[
	if (j < world.HEIGHT - 1 && world.At(i, j + 1).type == Tile::Type::Water)
	{
		if (erosion == 0)
		{
			world.At(i, j) = Sand(world.Clock() + 1);
			return;
		}
		else if (erosion > 0)
		{
			erosion--;
		}
	}
	clock++;
]
- Color TypeColor()
[
	return Color{ static_cast<unsigned char>(180 - Min(Max(erosion, 0), 10) * 10), static_cast<unsigned char>(180 - Min(Max(erosion, 0), 10) * 10), static_cast<unsigned char>(180 - Min(Max(erosion, 0), 10) * 10), 255 };
]
= int erosion

subclass Water implements Tile
- void Tick(int i, int j, World& world)
[
	if (j > 0)
	{
		if (world.At(i, j - 1).clock < 0)
		{
			world.At(i, j - 1) = Water(world.Clock() + 1);
			world.At(i, j) = Air(-1);
			return;
		}
		bool left = true;
		bool right = true;
		for (int offset = 0; offset < Max(i + 1, world.WIDTH - i); offset++)
		{
			if (left && i - offset >= 0) { left = world.At(i - offset, j - 1).type == Tile::Type::Water || world.At(i - offset, j - 1).clock < 0; }
			if (left && i - offset >= 0 && world.At(i - offset, j - 1).clock < 0)
			{
				world.At(i - offset, j - 1) = Water(world.Clock() + 1);
				world.At(i, j) = Air(-1);
				return;
			}
			if (right && i + offset < world.WIDTH) { right = world.At(i + offset, j - 1).type == Tile::Type::Water || world.At(i + offset, j - 1).clock < 0; }
			if (right && i + offset < world.WIDTH && world.At(i + offset, j - 1).clock < 0)
			{
				world.At(i + offset, j - 1) = Water(world.Clock() + 1);
				world.At(i, j) = Air(-1);
				return;
			}
		}
	}
	clock++;
]
- Color TypeColor()
[
	return BLUE;
]

subclass Plant implements Tile
- void Tick(int i, int j, World& world)
[
	if (i > 0)
	{
		switch (world.At(i - 1, j).type)
		{
		case Tile::Type::Water:
			if (energy < 100)
			{
				world.At(i - 1, j) = Air(-1);
				energy += 13;
			}
			break;

		case Tile::Type::Plant:
			if (rand() % 6 > 1 && energy > 5 && energy > world.At(i - 1, j).energy)
			{
				world.At(i - 1, j).energy += 4;
				energy -= 4;
			}
			break;
		}
	}
	if (i < world.WIDTH - 1)
	{
		switch (world.At(i + 1, j).type)
		{
		case Tile::Type::Water:
			if (energy < 100)
			{
				world.At(i + 1, j) = Air(-1);
				energy += 13;
			}
			break;

		case Tile::Type::Plant:
			if (rand() % 6 > 1 && energy > 5 && energy > world.At(i + 1, j).energy)
			{
				world.At(i + 1, j).energy += 4;
				energy -= 4;
			}
			break;
		}
	}
	if (j > 0)
	{
		switch (world.At(i, j - 1).type)
		{
		case Tile::Type::Water:
			if (energy < 100)
			{
				world.At(i, j - 1) = Air(-1);
				energy += 13;
			}
			break;

		case Tile::Type::Plant:
			if (rand() % 6 > 1 && energy > 5 && energy > world.At(i, j - 1).energy)
			{
				world.At(i, j - 1).energy += 4;
				energy -= 4;
			}
			break;
		}
	}
	if (j < world.HEIGHT - 1)
	{
		switch (world.At(i, j + 1).type)
		{
		case Tile::Type::Water:
			if (energy < 100)
			{
				world.At(i, j + 1) = Air(-1);
				energy += 13;
			}
			break;

		case Tile::Type::Plant:
			if (rand() % 6 > 1 && energy > 5 && energy > world.At(i, j + 1).energy)
			{
				world.At(i, j + 1).energy += 4;
				energy -= 4;
			}
			break;
		}
	}
	int neighbors = 0;
	for (int di = -1; di <= 1; di++)
	{
		for (int dj = -1; dj <= 1; dj++)
		{
			if (i + di >= 0 && i + di < world.WIDTH && j + dj >= 0 && j + dj < world.WIDTH && world.At(i + di, j + dj).type == Tile::Type::Plant)
			{
				neighbors++;
			}
		}
	}
	if (neighbors < 5 && energy >= 50)
	{
		switch (rand() % 4)
		{
		case 0:
			if (i > 0 && world.At(i - 1, j).clock < 0)
			{
				world.At(i - 1, j) = Plant(world.Clock(), 10);
				energy -= 50;
			}
			break;
			
		case 1:
			if (i < world.WIDTH - 1 && world.At(i + 1, j).clock < 0)
			{
				world.At(i + 1, j) = Plant(world.Clock(), 10);
				energy -= 50;
			}
			break;

		case 2:
			if (j > 0 && world.At(i, j - 1).clock < 0)
			{
				world.At(i, j - 1) = Plant(world.Clock(), 10);
				energy -= 50;
			}
			break;
			
		case 3:
			if (j < world.HEIGHT - 1 && world.At(i, j + 1).clock < 0)
			{
				world.At(i, j + 1) = Plant(world.Clock(), 10);
				energy -= 50;
			}
			break;
		}
	}
	if (energy < -10)
	{
		world.At(i, j) = Stone(world.Clock(), -1);
	}
	else
	{
		if (clock++ % (rand() % 51 + 50) == 0)
		{
			energy--;
		}
	}
]
- Color TypeColor()
[
	return Color{ 0, static_cast<unsigned char>(150 + Min(Max(energy, -10), 100)), 0, 255 };
]
= int energy

subclass Fire implements Tile
- void Tick(int i, int j, World& world)
[
	if (lifetime >= 0)
	{
		lifetime--;

		for (int di = -1; di <= 1; di++)
		{
			for (int dj = -1; dj <= 1; dj++)
			{
				if (i + di >= 0 && i + di < world.WIDTH && j + dj >= 0 && j + dj < world.WIDTH)
				{
					switch (world.At(i + di, j + dj).type)
					{
					case Tile::Type::Plant:
						lifetime += Min(75, Max(world.At(i + di, j + dj).data.Plant.energy, 15));
						world.At(i + di, j + dj) = Fire(world.Clock() + 1, lifetime / 2 - 1);
						break;

					case Tile::Type::Water:
						lifetime -= 5;
						break;
					}
				}
			}
		}

		if (lifetime % (rand() % 6 + 5) == 0)
		{
			int di = rand() % 3 - 1;
			int dj = rand() % 3;
			if (i + di >= 0 && i + di < world.WIDTH && j + dj >= 0 && j + dj < world.HEIGHT && world.At(i + di, j + dj).clock < 0)
			{
				world.At(i + di, j + dj) = Fire(world.Clock() + 1, lifetime / 2 - 1);
			}
		}
	}
	else
	{
		world.At(i, j) = Air(-1);
	}
]
- Color TypeColor()
[
	return Color{ static_cast<unsigned char>(250 - Min(Max(0, lifetime), 50)), static_cast<unsigned char>(170 - Min(Max(0, lifetime), 50) * 2), 20, 255 };
]
= int lifetime
```
### Example output
```
struct Tile;

union TileData
{
	struct BaseData
	{
		long clock;
	} Base;
	struct AirData
	{
		long clock;
	} Air;
	struct SandData
	{
		long clock;
	} Sand;
	struct StoneData
	{
		long clock;
		int erosion;
	} Stone;
	struct WaterData
	{
		long clock;
	} Water;
	struct PlantData
	{
		long clock;
		int energy;
	} Plant;
	struct FireData
	{
		long clock;
		int lifetime
;
	} Fire;
	TileData() { }
	TileData(AirData data) : Air(data) { }
	TileData(SandData data) : Sand(data) { }
	TileData(StoneData data) : Stone(data) { }
	TileData(WaterData data) : Water(data) { }
	TileData(PlantData data) : Plant(data) { }
	TileData(FireData data) : Fire(data) { }
};

void AirTick(int i, int j, World& world, TileData& data, Tile& obj);
Color AirTypeColor(TileData& data, Tile& obj);

void SandTick(int i, int j, World& world, TileData& data, Tile& obj);
Color SandTypeColor(TileData& data, Tile& obj);

void StoneTick(int i, int j, World& world, TileData& data, Tile& obj);
Color StoneTypeColor(TileData& data, Tile& obj);

void WaterTick(int i, int j, World& world, TileData& data, Tile& obj);
Color WaterTypeColor(TileData& data, Tile& obj);

void PlantTick(int i, int j, World& world, TileData& data, Tile& obj);
Color PlantTypeColor(TileData& data, Tile& obj);

void FireTick(int i, int j, World& world, TileData& data, Tile& obj);
Color FireTypeColor(TileData& data, Tile& obj);

struct Tile
{
	enum class Type
	{
		Invalid,
		Air,
		Sand,
		Stone,
		Water,
		Plant,
		Fire,
	} type;

	TileData data;
	Tile() : type(Type::Invalid), data() { }
	Tile(Type type, TileData data) : type(type), data(data) { }

	void Tick(int i, int j, World& world)
	{
		switch (type)
		{
		case Type::Air:
			return AirTick(i, j, world, data, *this);
		case Type::Sand:
			return SandTick(i, j, world, data, *this);
		case Type::Stone:
			return StoneTick(i, j, world, data, *this);
		case Type::Water:
			return WaterTick(i, j, world, data, *this);
		case Type::Plant:
			return PlantTick(i, j, world, data, *this);
		case Type::Fire:
			return FireTick(i, j, world, data, *this);
		default:
			return void();
		}
	}

	Color TypeColor()
	{
		switch (type)
		{
		case Type::Air:
			return AirTypeColor(data, *this);
		case Type::Sand:
			return SandTypeColor(data, *this);
		case Type::Stone:
			return StoneTypeColor(data, *this);
		case Type::Water:
			return WaterTypeColor(data, *this);
		case Type::Plant:
			return PlantTypeColor(data, *this);
		case Type::Fire:
			return FireTypeColor(data, *this);
		default:
			return Color();
		}
	}
};

Tile Air(long clock);
Tile Sand(long clock);
Tile Stone(long clock, int erosion);
Tile Water(long clock);
Tile Plant(long clock, int energy);
Tile Fire(long clock, int lifetime
);

void AirTick(int i, int j, World& world, TileData& data, Tile& obj)
{
	data.Air.clock = -1;
}
Color AirTypeColor(TileData& data, Tile& obj)
{
	return BLANK;
}

void SandTick(int i, int j, World& world, TileData& data, Tile& obj)
{
	if (j > 0)
	{
		if (world.At(i, j - 1).data.Sand.clock < 0 || world.At(i, j - 1).type == Tile::Type::Fire)
		{
			world.At(i, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = Air(-1);
			return;
		}
		else if (world.At(i, j - 1).type == Tile::Type::Water)
		{
			world.At(i, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = Water(world.Clock() + 1);
			return;
		}
		else if (i > 0 && (world.At(i - 1, j).data.Sand.clock < 0 || world.At(i - 1, j).type == Tile::Type::Fire) && (world.At(i - 1, j - 1).data.Sand.clock < 0 || world.At(i - 1, j - 1).type == Tile::Type::Fire))
		{
			world.At(i - 1, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = Air(-1);
			return;
		}
		else if (i > 0 && (world.At(i - 1, j).data.Sand.clock < 0 || world.At(i - 1, j).type == Tile::Type::Water || world.At(i - 1, j).type == Tile::Type::Fire) && world.At(i - 1, j - 1).type == Tile::Type::Water)
		{
			world.At(i - 1, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = (world.At(i - 1, j).data.Sand.clock < 0 ? Air(-1) : Water(world.Clock() + 1));
			world.At(i - 1, j) = Water(world.Clock() + 1);
			return;
		}
		else if (i < world.WIDTH - 1 && (world.At(i + 1, j).data.Sand.clock < 0 || world.At(i + 1, j).type == Tile::Type::Fire) && (world.At(i + 1, j - 1).data.Sand.clock < 0 || world.At(i + 1, j - 1).type == Tile::Type::Fire))
		{
			world.At(i + 1, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = Air(-1);
			return;
		}
		else if (i < world.WIDTH - 1 && (world.At(i + 1, j).data.Sand.clock < 0 || world.At(i + 1, j).type == Tile::Type::Water || world.At(i + 1, j).type == Tile::Type::Fire) && world.At(i + 1, j - 1).type == Tile::Type::Water)
		{
			world.At(i + 1, j - 1) = Sand(world.Clock() + 1);
			world.At(i, j) = (world.At(i + 1, j).data.Sand.clock < 0 ? Air(-1) : Water(world.Clock() + 1));
			world.At(i + 1, j) = Water(world.Clock() + 1);
			return;
		}
	}
	data.Sand.clock++;
}
Color SandTypeColor(TileData& data, Tile& obj)
{
	return Color{ 237, 201, 175, 255 };
}

void StoneTick(int i, int j, World& world, TileData& data, Tile& obj)
{
	if (j < world.HEIGHT - 1 && world.At(i, j + 1).type == Tile::Type::Water)
	{
		if (data.Stone.erosion == 0)
		{
			world.At(i, j) = Sand(world.Clock() + 1);
			return;
		}
		else if (data.Stone.erosion > 0)
		{
			data.Stone.erosion--;
		}
	}
	data.Stone.clock++;
}
Color StoneTypeColor(TileData& data, Tile& obj)
{
	return Color{ static_cast<unsigned char>(180 - Min(Max(data.Stone.erosion, 0), 10) * 10), static_cast<unsigned char>(180 - Min(Max(data.Stone.erosion, 0), 10) * 10), static_cast<unsigned char>(180 - Min(Max(data.Stone.erosion, 0), 10) * 10), 255 };
}

void WaterTick(int i, int j, World& world, TileData& data, Tile& obj)
{
	if (j > 0)
	{
		if (world.At(i, j - 1).data.Water.clock < 0)
		{
			world.At(i, j - 1) = Water(world.Clock() + 1);
			world.At(i, j) = Air(-1);
			return;
		}
		bool left = true;
		bool right = true;
		for (int offset = 0; offset < Max(i + 1, world.WIDTH - i); offset++)
		{
			if (left && i - offset >= 0) { left = world.At(i - offset, j - 1).type == Tile::Type::Water || world.At(i - offset, j - 1).data.Water.clock < 0; }
			if (left && i - offset >= 0 && world.At(i - offset, j - 1).data.Water.clock < 0)
			{
				world.At(i - offset, j - 1) = Water(world.Clock() + 1);
				world.At(i, j) = Air(-1);
				return;
			}
			if (right && i + offset < world.WIDTH) { right = world.At(i + offset, j - 1).type == Tile::Type::Water || world.At(i + offset, j - 1).data.Water.clock < 0; }
			if (right && i + offset < world.WIDTH && world.At(i + offset, j - 1).data.Water.clock < 0)
			{
				world.At(i + offset, j - 1) = Water(world.Clock() + 1);
				world.At(i, j) = Air(-1);
				return;
			}
		}
	}
	data.Water.clock++;
}
Color WaterTypeColor(TileData& data, Tile& obj)
{
	return BLUE;
}

void PlantTick(int i, int j, World& world, TileData& data, Tile& obj)
{
	if (i > 0)
	{
		switch (world.At(i - 1, j).type)
		{
		case Tile::Type::Water:
			if (data.Plant.energy < 100)
			{
				world.At(i - 1, j) = Air(-1);
				data.Plant.energy += 13;
			}
			break;

		case Tile::Type::Plant:
			if (rand() % 6 > 1 && data.Plant.energy > 5 && data.Plant.energy > world.At(i - 1, j).data.Plant.energy)
			{
				world.At(i - 1, j).data.Plant.energy += 4;
				data.Plant.energy -= 4;
			}
			break;
		}
	}
	if (i < world.WIDTH - 1)
	{
		switch (world.At(i + 1, j).type)
		{
		case Tile::Type::Water:
			if (data.Plant.energy < 100)
			{
				world.At(i + 1, j) = Air(-1);
				data.Plant.energy += 13;
			}
			break;

		case Tile::Type::Plant:
			if (rand() % 6 > 1 && data.Plant.energy > 5 && data.Plant.energy > world.At(i + 1, j).data.Plant.energy)
			{
				world.At(i + 1, j).data.Plant.energy += 4;
				data.Plant.energy -= 4;
			}
			break;
		}
	}
	if (j > 0)
	{
		switch (world.At(i, j - 1).type)
		{
		case Tile::Type::Water:
			if (data.Plant.energy < 100)
			{
				world.At(i, j - 1) = Air(-1);
				data.Plant.energy += 13;
			}
			break;

		case Tile::Type::Plant:
			if (rand() % 6 > 1 && data.Plant.energy > 5 && data.Plant.energy > world.At(i, j - 1).data.Plant.energy)
			{
				world.At(i, j - 1).data.Plant.energy += 4;
				data.Plant.energy -= 4;
			}
			break;
		}
	}
	if (j < world.HEIGHT - 1)
	{
		switch (world.At(i, j + 1).type)
		{
		case Tile::Type::Water:
			if (data.Plant.energy < 100)
			{
				world.At(i, j + 1) = Air(-1);
				data.Plant.energy += 13;
			}
			break;

		case Tile::Type::Plant:
			if (rand() % 6 > 1 && data.Plant.energy > 5 && data.Plant.energy > world.At(i, j + 1).data.Plant.energy)
			{
				world.At(i, j + 1).data.Plant.energy += 4;
				data.Plant.energy -= 4;
			}
			break;
		}
	}
	int neighbors = 0;
	for (int di = -1; di <= 1; di++)
	{
		for (int dj = -1; dj <= 1; dj++)
		{
			if (i + di >= 0 && i + di < world.WIDTH && j + dj >= 0 && j + dj < world.WIDTH && world.At(i + di, j + dj).type == Tile::Type::Plant)
			{
				neighbors++;
			}
		}
	}
	if (neighbors < 5 && data.Plant.energy >= 50)
	{
		switch (rand() % 4)
		{
		case 0:
			if (i > 0 && world.At(i - 1, j).data.Plant.clock < 0)
			{
				world.At(i - 1, j) = Plant(world.Clock(), 10);
				data.Plant.energy -= 50;
			}
			break;
			
		case 1:
			if (i < world.WIDTH - 1 && world.At(i + 1, j).data.Plant.clock < 0)
			{
				world.At(i + 1, j) = Plant(world.Clock(), 10);
				data.Plant.energy -= 50;
			}
			break;

		case 2:
			if (j > 0 && world.At(i, j - 1).data.Plant.clock < 0)
			{
				world.At(i, j - 1) = Plant(world.Clock(), 10);
				data.Plant.energy -= 50;
			}
			break;
			
		case 3:
			if (j < world.HEIGHT - 1 && world.At(i, j + 1).data.Plant.clock < 0)
			{
				world.At(i, j + 1) = Plant(world.Clock(), 10);
				data.Plant.energy -= 50;
			}
			break;
		}
	}
	if (data.Plant.energy < -10)
	{
		world.At(i, j) = Stone(world.Clock(), -1);
	}
	else
	{
		if (data.Plant.clock++ % (rand() % 51 + 50) == 0)
		{
			data.Plant.energy--;
		}
	}
}
Color PlantTypeColor(TileData& data, Tile& obj)
{
	return Color{ 0, static_cast<unsigned char>(150 + Min(Max(data.Plant.energy, -10), 100)), 0, 255 };
}

void FireTick(int i, int j, World& world, TileData& data, Tile& obj)
{
	if (lifetime >= 0)
	{
		lifetime--;

		for (int di = -1; di <= 1; di++)
		{
			for (int dj = -1; dj <= 1; dj++)
			{
				if (i + di >= 0 && i + di < world.WIDTH && j + dj >= 0 && j + dj < world.WIDTH)
				{
					switch (world.At(i + di, j + dj).type)
					{
					case Tile::Type::Plant:
						lifetime += Min(75, Max(world.At(i + di, j + dj).data.Plant.energy, 15));
						world.At(i + di, j + dj) = Fire(world.Clock() + 1, lifetime / 2 - 1);
						break;

					case Tile::Type::Water:
						lifetime -= 5;
						break;
					}
				}
			}
		}

		if (lifetime % (rand() % 6 + 5) == 0)
		{
			int di = rand() % 3 - 1;
			int dj = rand() % 3;
			if (i + di >= 0 && i + di < world.WIDTH && j + dj >= 0 && j + dj < world.HEIGHT && world.At(i + di, j + dj).data.Fire.clock < 0)
			{
				world.At(i + di, j + dj) = Fire(world.Clock() + 1, lifetime / 2 - 1);
			}
		}
	}
	else
	{
		world.At(i, j) = Air(-1);
	}
}
Color FireTypeColor(TileData& data, Tile& obj)
{
	return Color{ static_cast<unsigned char>(250 - Min(Max(0, lifetime), 50)), static_cast<unsigned char>(170 - Min(Max(0, lifetime), 50) * 2), 20, 255 };
}

Tile Air(long clock)
{
	TileData::AirData data;
	data.clock = clock;
	return Tile(Tile::Type::Air, data);
}
Tile Sand(long clock)
{
	TileData::SandData data;
	data.clock = clock;
	return Tile(Tile::Type::Sand, data);
}
Tile Stone(long clock, int erosion)
{
	TileData::StoneData data;
	data.clock = clock;
	data.erosion = erosion;
	return Tile(Tile::Type::Stone, data);
}
Tile Water(long clock)
{
	TileData::WaterData data;
	data.clock = clock;
	return Tile(Tile::Type::Water, data);
}
Tile Plant(long clock, int energy)
{
	TileData::PlantData data;
	data.clock = clock;
	data.energy = energy;
	return Tile(Tile::Type::Plant, data);
}
Tile Fire(long clock, int lifetime
)
{
	TileData::FireData data;
	data.clock = clock;
	data.lifetime
 = lifetime
;
	return Tile(Tile::Type::Fire, data);
}
```

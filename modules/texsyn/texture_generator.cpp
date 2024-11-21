#include "texture_generator.h"
#include <map>

namespace TexSyn
{

TextureGenerator::TextureGenerator() :
	RefCounted(),
	m_image(),
	m_width(512),
	m_height(512)
{}

void TextureGenerator::setWidth(int width)
{
	m_width = width;
}

void TextureGenerator::setHeight(int height)
{
	m_height = height;
}

void TextureGenerator::generateRandomGradiantGrid(int cellSize)
{
	m_image.init(m_width, m_height, 3, true);
	RandomNumberGenerator rng;
	for(int y=0; y<m_height; ++y)
	{
		for(int x=0; x<m_width; ++x)
		{
			//get current cell Index
			int cellX = x / cellSize;
			int cellY = y / cellSize;
			int cellIndex = cellY * (m_width / cellSize) + cellX;

			//Get random gradient
			Gradient gradient = getRandomGradient(cellIndex);
			
			rng.set_seed(cellIndex);
			float gradientX = rng.randf();
			float gradientY = rng.randf();
			
			//Project x and y onto cell coordinates
			float xF = float(x % cellSize) / cellSize;
			float yF = float(y % cellSize) / cellSize;
			
			Vector2 gradientVector(gradientX, gradientY);
			Vector2 positionVector(xF, yF);
			float dot = gradientVector.dot(positionVector);
			dot = CLAMP(dot, 0.0f, 1.0f);
			
			Color finalColor = gradient.color1.lerp(gradient.color2, dot);
			m_image.set_pixelColor(x, y, finalColor);
		}
	}
}

void TextureGenerator::generateGridIDMap(int cellSize)
{
	m_image.init(m_width, m_height, 3, true);
	RandomNumberGenerator rng;
	for(int y=0; y<m_height; ++y)
	{
		for(int x=0; x<m_width; ++x)
		{
			//get current cell Index
			int cellX = x / cellSize;
			int cellY = y / cellSize;
			int cellIndex = cellY * (m_width / cellSize) + cellX;
			
			rng.set_seed(cellIndex);
			
			//Get random green and blue values
			float mean = 0.5;
			float variance = 1.0/6.0;
			float green = rng.randfn(mean, variance);
			float blue = rng.randfn(mean, variance);
			
			int cellCount = (m_width / cellSize) * (m_height / cellSize);
			
			//get color from cell index
			Color finalColor = Color(float(cellIndex) / cellCount, green, blue);
			
			//set pixel
			m_image.set_pixelColor(x, y, finalColor);
		}
	}
}

void TextureGenerator::generateRandomGradientCellular(int cellSize)
{
	m_image.init(m_width, m_height, 3, true);
	RandomNumberGenerator rng;
	std::map<Vector2i, Vector2> coresMap;
	std::map<Vector2i, Gradient> gradientsMap;
	
	//Create gradients and cores for each cell
	for(int y=0; y<m_height; y+=cellSize)
	{
		for(int x=0; x<m_width; x+=cellSize)
		{
			//get current cell Index
			Vector2i cellIndex2i(x/cellSize, y/cellSize);
			int cellIndex = cellIndex2i.y * m_width/cellSize + cellIndex2i.x;

			rng.set_seed(cellIndex);
			
			Vector2 core;
			core.x = rng.randf_range(0.0, 1.0);
			core.y = rng.randf_range(0.0, 1.0);
			coresMap.insert(std::make_pair(cellIndex2i, core));
			gradientsMap.insert(std::make_pair(cellIndex2i, getRandomGradient(cellIndex)));
		}
	}
	
	//Fill the texture
	for(int y=0; y<m_height; ++y)
	{
		for(int x=0; x<m_width; ++x)
		{
			//get closest core
			Vector2i closestCoreCell;
			float closestDistance = 9999999.0f;
			Vector2 periodicityOffset(0, 0);
			for(int y2 = -1; y2 <= 1; ++y2)
			{
				for(int x2 = -1; x2 <= 1; ++x2)
				{
					Vector2i cellIndex2iNonPeriodic(x/cellSize + x2, y/cellSize + y2);
					Vector2i cellIndex2i(cellIndex2iNonPeriodic);
					
					//Periodicity
					if(cellIndex2i.x < 0)
						cellIndex2i.x = m_width/cellSize - 1;
					if(cellIndex2i.x >= m_width/cellSize)
						cellIndex2i.x = 0;
					if(cellIndex2i.y < 0)
						cellIndex2i.y = m_height/cellSize - 1;
					if (cellIndex2i.y >= m_height/cellSize)
						cellIndex2i.y = 0;
					
					if(coresMap.find(cellIndex2i) != coresMap.end())
					{
						Vector2 coreAbsolutePos = coresMap[cellIndex2i]*cellSize + Vector2(cellIndex2i.x*cellSize, cellIndex2i.y*cellSize);
						Vector2 tempPeriodicityOffset(0, 0);
						
						if(cellIndex2i.x < cellIndex2iNonPeriodic.x)
						{
							coreAbsolutePos.x += m_width;
							tempPeriodicityOffset.x = +m_width;
						}
						if(cellIndex2i.x > cellIndex2iNonPeriodic.x)
						{
							coreAbsolutePos.x -= m_width;
							tempPeriodicityOffset.x = -m_width;
						}
						if(cellIndex2i.y < cellIndex2iNonPeriodic.y)
						{
							coreAbsolutePos.y += m_height;
							tempPeriodicityOffset.y = +m_height;
						}
						if(cellIndex2i.y > cellIndex2iNonPeriodic.y)
						{
							coreAbsolutePos.y -= m_height;
							tempPeriodicityOffset.y = -m_height;
						}
						
						Vector2 position(x, y);
						float distance = coreAbsolutePos.distance_to(position);
						if(distance < closestDistance)
						{
							closestCoreCell = cellIndex2i;
							closestDistance = distance;
							periodicityOffset = tempPeriodicityOffset;
						}
					}
				}
			}
			
			//get gradient
			Vector2 coreAbsolutePos = coresMap[closestCoreCell] + Vector2(closestCoreCell.x*cellSize, closestCoreCell.y*cellSize) + periodicityOffset;
			Gradient gradient = gradientsMap[closestCoreCell];
			int cellIndex = closestCoreCell.y * (m_width / cellSize) + closestCoreCell.x;
			
			rng.set_seed(cellIndex);
			float gradientX = rng.randf()*2.0 - 1.0;
			float gradientY = rng.randf()*2.0 - 1.0;
			
			//Project x and y onto cell coordinates
			float xF = (float(x) - coreAbsolutePos.x)/cellSize;
			float yF = (float(y) - coreAbsolutePos.y)/cellSize;
			
			Vector2 gradientVector(gradientX, gradientY);
			Vector2 positionVector(xF, yF);
			float dot = gradientVector.dot(positionVector);
			dot = CLAMP(dot, 0.0f, 1.0f);
			
			Color finalColor = gradient.color1.lerp(gradient.color2, dot);
			m_image.set_pixelColor(x, y, finalColor);
		}
	}
}

void TextureGenerator::generateCellularIDMap(int cellSize)
{
	m_image.init(m_width, m_height, 3, true);
	RandomNumberGenerator rng;
	std::map<Vector2i, Vector2> coresMap;
	for(int y=0; y<m_height; y+=cellSize)
	{
		for(int x=0; x<m_width; x+=cellSize)
		{
			//get current cell Index
			Vector2i cellIndex2i(x/cellSize, y/cellSize);
			int cellIndex = cellIndex2i.y * (m_width / cellSize) + cellIndex2i.x;

			rng.set_seed(cellIndex);
			
			Vector2 core;
			core.x = rng.randf_range(0.0, 1.0);
			core.y = rng.randf_range(0.0, 1.0);
			coresMap.insert(std::make_pair(cellIndex2i, core));
		}
	}
	
	for(int y=0; y<m_height; ++y)
	{
		for(int x=0; x<m_width; ++x)
		{
			Vector2 position(x, y);
			//get closest core
			Vector2i closestCoreCell;
			float closestDistance = 9999999.0f;
			for(int y2 = -1; y2 <= 1; ++y2)
			{
				for(int x2 = -1; x2 <= 1; ++x2)
				{
					Vector2i cellIndex2iNonPeriodic(x/cellSize + x2, y/cellSize + y2);
					Vector2i cellIndex2i(cellIndex2iNonPeriodic);
					
					//Periodicity
					if(cellIndex2i.x < 0)
						cellIndex2i.x = m_width/cellSize - 1;
					if(cellIndex2i.x >= m_width/cellSize)
						cellIndex2i.x = 0;
					if(cellIndex2i.y < 0)
						cellIndex2i.y = m_height/cellSize - 1;
					if (cellIndex2i.y >= m_height/cellSize)
						cellIndex2i.y = 0;
					
					if(coresMap.find(cellIndex2i) != coresMap.end())
					{
						Vector2 coreAbsolutePos = coresMap[cellIndex2i]*cellSize + Vector2(cellIndex2i.x*cellSize, cellIndex2i.y*cellSize);
						
						if(cellIndex2i.x < cellIndex2iNonPeriodic.x)
						{
							coreAbsolutePos.x += m_width;
						}
						if(cellIndex2i.x > cellIndex2iNonPeriodic.x)
						{
							coreAbsolutePos.x -= m_width;
						}
						if(cellIndex2i.y < cellIndex2iNonPeriodic.y)
						{
							coreAbsolutePos.y += m_height;
						}
						if(cellIndex2i.y > cellIndex2iNonPeriodic.y)
						{
							coreAbsolutePos.y -= m_height;
						}
						
						float distance = coreAbsolutePos.distance_to(position);
						if(distance < closestDistance)
						{
							closestCoreCell = cellIndex2i;
							closestDistance = distance;
						}
					}
				}
			}
			
			int cellIndex = closestCoreCell.y * (m_width / cellSize) + closestCoreCell.x;
			rng.set_seed(cellIndex);
			
			//Get random green and blue values
			float mean = 0.5;
			float variance = 1.0/6.0;
			float green = rng.randfn(mean, variance);
			float blue = rng.randfn(mean, variance);
			
			int cellCount = (m_width / cellSize) * (m_height / cellSize);
			
			//get color from cell index
			Color finalColor = Color(float(cellIndex) / cellCount, green, blue);
			
			//set pixel
			m_image.set_pixelColor(x, y, finalColor);
		}
	}
}

void TextureGenerator::toImage(Ref<Image> refImage) const
{
	ERR_FAIL_COND_MSG(refImage.is_null(), "image must not be a null reference.");
	Ref<Image> tmpResultRef;
	tmpResultRef = Image::create_empty(m_image.get_width(), m_image.get_height(), false, Image::FORMAT_RGBF);
	m_image.toImageIndexed(tmpResultRef, 0);
	refImage->copy_internals_from(tmpResultRef);
}

TextureGenerator::Gradient TextureGenerator::getRandomGradient(int seed) const
{
	Gradient gradient;
	RandomNumberGenerator rng;
	rng.set_seed(seed);
	float mean = 0.5;
	float variance = 1.0/6.0;
	gradient.color1 = Color(rng.randfn(mean, variance), rng.randfn(mean, variance), rng.randfn(mean, variance));
	gradient.color2 = Color(rng.randfn(mean, variance), rng.randfn(mean, variance), rng.randfn(mean, variance));
	return gradient;
}

void TextureGenerator::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("setWidth", "width"), &TextureGenerator::setWidth);
	ClassDB::bind_method(D_METHOD("setHeight", "height"), &TextureGenerator::setHeight);
	ClassDB::bind_method(D_METHOD("generateRandomGradiantGrid", "cellSize"), &TextureGenerator::generateRandomGradiantGrid);
	ClassDB::bind_method(D_METHOD("generateGridIDMap", "cellSize"), &TextureGenerator::generateGridIDMap);
	ClassDB::bind_method(D_METHOD("generateRandomGradientCellular", "cellSize"), &TextureGenerator::generateRandomGradientCellular);
	ClassDB::bind_method(D_METHOD("generateCellularIDMap", "cellSize"), &TextureGenerator::generateCellularIDMap);
	
	ClassDB::bind_method(D_METHOD("toImage", "refImage"), &TextureGenerator::toImage);
}

}

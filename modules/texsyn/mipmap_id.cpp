#include "mipmap_id.h"

namespace TexSyn
{

Mipmap::Mipmap() :
	m_mipmap()
{}

void Mipmap::setTexture(const ImageType &image)
{
	m_mipmap.resize(0);
	m_mipmap.push_back(image);
}

void Mipmap::computeMipmap()
{
	DEV_ASSERT(m_mipmap.size() > 0);
	unsigned int i = 0;
	unsigned int width = m_mipmap[0].get_width();
	unsigned int height = m_mipmap[0].get_height();
	unsigned int d = m_mipmap[0].get_nbDimensions();
	while(width>1 || height>1)
	{
		++i;
		width/=2;
		height/=2;
		ImageType map;
		map.init(width, height, d, true);
		map.for_all_images([&] (ImageType::ImageScalarType &image, unsigned int d)
		{
			image.for_all_pixels([&] (ImageType::DataType &pix, int x, int y)
			{
				ImageType::DataType pix00, pix01, pix10, pix11;
				pix00 = m_mipmap[i-1].get_image(d).get_pixel(x*2, y*2);
				pix01 = m_mipmap[i-1].get_image(d).get_pixel(x*2, y*2+1);
				pix10 = m_mipmap[i-1].get_image(d).get_pixel(x*2+1, y*2);
				pix11 = m_mipmap[i-1].get_image(d).get_pixel(x*2+1, y*2+1);
				pix = (pix00 + pix01 + pix10 + pix11)/4.0;
			});
		});
		m_mipmap.push_back(map);
	}
}

void Mipmap::upsizeMipmap()
{
	DEV_ASSERT(m_mipmap.size() > 0);
	unsigned int width = m_mipmap[0].get_width();
	unsigned int height = m_mipmap[0].get_height();
	unsigned int d = m_mipmap[0].get_nbDimensions();
	for(unsigned int i=1; i<m_mipmap.size(); ++i)
	{
		ImageType resizedMap;
		resizedMap.init(width, height, d, true);
		int factor = int(pow(2.0, float(i)));
		resizedMap.for_all_images([&] (ImageType::ImageScalarType &image, unsigned int d)
		{
			image.for_all_pixels([&] (ImageType::DataType &pix, int x, int y)
			{
				pix = m_mipmap[i].get_image(d).get_pixel(x/factor, y/factor);
			});
		});

		m_mipmap[i] = resizedMap;
	}
}

const Mipmap::ImageType &Mipmap::mipmap(int i) const
{
	return m_mipmap[i];
}

int Mipmap::nbMaps() const
{
	return m_mipmap.size();
}

MipmapMultiIDMap::MipmapMultiIDMap() :
	m_mipmap()
{}

void MipmapMultiIDMap::setIDMap(const ImageMultiIDMapType &idMap)
{
	m_mipmap.resize(0);
	m_mipmap.push_back(idMap);
}

void MipmapMultiIDMap::computeMipmap()
{
	DEV_ASSERT(m_mipmap.size() > 0);
	unsigned int i = 0;
	unsigned width = m_mipmap[0].get_width();
	unsigned height = m_mipmap[0].get_height();
	while(width>1 || height>1)
	{
		++i;
		width/=2;
		height/=2;
		ImageMultiIDMapType map;
		map.init(width, height, true);
		map.for_all_pixels([&] (ImageMultiIDMapType::DataType &pix, int x, int y)
		{
			ImageMultiIDMapType::DataType pix00, pix01, pix10, pix11;
			pix00 = m_mipmap[i-1].get_pixel(x*2, y*2);
			pix01 = m_mipmap[i-1].get_pixel(x*2, y*2+1);
			pix10 = m_mipmap[i-1].get_pixel(x*2+1, y*2);
			pix11 = m_mipmap[i-1].get_pixel(x*2+1, y*2+1);
			pix |= pix00;
			pix |= pix01;
			pix |= pix10;
			pix |= pix11;
		});
		m_mipmap.push_back(map);
	}
}

void MipmapMultiIDMap::upsizeMipmap()
{
	DEV_ASSERT(m_mipmap.size() > 0);
	unsigned width = m_mipmap[0].get_width();
	unsigned height = m_mipmap[0].get_height();
	for(unsigned int i=1; i<m_mipmap.size(); ++i)
	{
		ImageMultiIDMapType resizedMap;
		resizedMap.init(width, height, true);
		int factor = int(pow(2.0, float(i)));
		resizedMap.for_all_pixels([&] (ImageMultiIDMapType::DataType &pix, int x, int y)
		{
			pix = m_mipmap[i].get_pixel(x/factor, y/factor);
		});
		m_mipmap[i] = resizedMap;
	}
}

const MipmapMultiIDMap::ImageMultiIDMapType &MipmapMultiIDMap::mipmap(int i) const
{
	return m_mipmap[i];
}

int MipmapMultiIDMap::nbMaps() const
{
	return m_mipmap.size();
}

};

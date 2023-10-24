#include "mipmaps.h"

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

Mipmap::ImageType &Mipmap::mipmap(int i)
{
	return m_mipmap[i];
}

int Mipmap::nbMaps() const
{
	return m_mipmap.size();
}

void Mipmap::toImage(Ref<Image> image, Image::Format format)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(m_mipmap.size() == 0, "mipmap must be set.");
	//TODO: check if mipmap was not resized
	
	if(format == Image::FORMAT_MAX)
	{
		format = image->get_format();
	}
	
	bool use_mipmaps = m_mipmap.size() > 1;
	
	Vector<uint8_t> fullData;
	for(unsigned int i=0; i<m_mipmap.size(); ++i)
	{
		const ImageType &texsynImage = m_mipmap[i];
		Ref<Image> gdImage = Image::create_empty(texsynImage.get_width(), texsynImage.get_height(), false, format);
		texsynImage.toImage(gdImage);
		Vector<uint8_t> data = gdImage->get_data();
		for(int j=0; j<data.size(); ++j)
		{
			fullData.push_back(data[j]);
		}
	}
	image->set_data(m_mipmap[0].get_width(), m_mipmap[0].get_height(), use_mipmaps, format, fullData);
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

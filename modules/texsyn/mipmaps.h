#ifndef TEXSYN_MIPMAP_ID_H
#define TEXSYN_MIPMAP_ID_H

#include "image_vector.h"

namespace TexSyn
{

struct BitVector
{
	uint64_t lo;
	uint64_t hi;
	
	BitVector &operator &=(const BitVector &other)
	{
		lo &= other.lo;
		hi &= other.hi;
		return *this;
	}
	
	BitVector &operator |=(const BitVector &other)
	{
		lo |= other.lo;
		hi |= other.hi;
		return *this;
	}
	
	BitVector &operator &=(int i)
	{
		if(i<64)
		{
			lo = uint64_t(1) << i & lo;
			hi = 0;
		}
		else
		{
			lo = 0;
			hi = uint64_t(1) << (i-64) & hi;
		}
		return *this;
	}
	
	BitVector &operator |=(int i)
	{
		if(i<64)
		{
			lo = uint64_t(1) << i | lo;
		}
		else
		{
			hi = uint64_t(1) << (i-64) | hi;
		}
		return *this;
	}
	
	BitVector operator&(const BitVector &other) const
	{
		BitVector bv;
		bv = *this;
		bv &= other;
		return bv;
	}
	
	BitVector operator|(const BitVector &other) const
	{
		BitVector bv;
		bv = *this;
		bv |= other;
		return bv;
	}
	
	BitVector operator&(int i) const
	{
		BitVector bv;
		bv = *this;
		bv &= i;
		return bv;
	}
	
	BitVector operator|(int i) const
	{
		BitVector bv;
		bv = *this;
		bv |= i;
		return bv;
	}
	
	bool toBool() const
	{
		return lo || hi;
	}
	
	int log2() const
	{
		const int tab64[64] = {
		63,  0, 58,  1, 59, 47, 53,  2,
		60, 39, 48, 27, 54, 33, 42,  3,
		61, 51, 37, 40, 49, 18, 28, 20,
		55, 30, 34, 11, 43, 14, 22,  4,
		62, 57, 46, 52, 38, 26, 32, 41,
		50, 36, 17, 19, 29, 10, 13, 21,
		56, 45, 25, 31, 35, 16,  9, 12,
		44, 24, 15,  8, 23,  7,  6,  5};
		
		int result = 0;
		uint64_t value;
		if(hi>0)
		{
			result += 64;
			value = hi;
		}
		else
		{
			value = lo;
		}
		value |= value >> 1;
		value |= value >> 2;
		value |= value >> 4;
		value |= value >> 8;
		value |= value >> 16;
		value |= value >> 32;
		return result + tab64[((uint64_t)((value - (value >> 1))*0x07EDD5E59A4E28C2)) >> 58];
	}
};

class Mipmap
{
public:
	using ImageType = ImageVector<float>;
	using ImageArrayType=LocalVector<ImageType>;

	Mipmap();
	
	void setTexture(const ImageType &image);
	void computeMipmap();
	void upsizeMipmap();
	const ImageType &mipmap(int i) const;
	ImageType &mipmap(int i);
	int nbMaps() const;
	
	void toImage(Ref<Image> image, Image::Format format = Image::FORMAT_MAX);
	
private:
	ImageArrayType m_mipmap;
};

class MipmapMultiIDMap
{
public:
	using ImageMultiIDMapType = ImageScalar<BitVector>;
	using ImageArrayType=LocalVector<ImageMultiIDMapType>;
	
	MipmapMultiIDMap();
	
	void setIDMap(const ImageMultiIDMapType &idMap);
	void computeMipmap();
	void upsizeMipmap();
	const ImageMultiIDMapType &mipmap(int i) const;
	int nbMaps() const;

private:
	ImageArrayType m_mipmap;
};

};

#endif

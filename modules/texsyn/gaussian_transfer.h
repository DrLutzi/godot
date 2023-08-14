#ifndef TEXSYN_GAUSSIAN_TRANSFER_H
#define TEXSYN_GAUSSIAN_TRANSFER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "image_vector.h"
#include "mipmap_id.h"

namespace TexSyn {

class GaussianTransfer
{
public:
	using ImageType = ImageVector<double>;
	using DataType = typename ImageType::DataType;
	using VectorType = typename ImageType::VectorType;
	using ImageRegionMapType = ImageScalar<int>;
	using ImageMultipleRegionMapType = ImageScalar<BitVector>;

private:
	struct PixelSortStruct
	{
		int x;
		int y;
		DataType value;
		bool operator < (const PixelSortStruct& other) const
		{
			return (value < other.value);
		}
	};

	DataType m_mean;
	DataType m_variance;

public:
	static double Erf(double x);
	static double ErfInv(double x);
	static double CDF(double x, double mu, double sigma);
	static double invCDF(double U, double mu, double sigma);
	
	void toMultipleRegions(ImageMultipleRegionMapType &imageMultipleRegions, const ImageRegionMapType &imageRegion);
	
	GaussianTransfer();
	void setMean(DataType mean);
	void setVariance(DataType variance);
	void computeTinput(const ImageType &input, ImageType &T_input, bool do_clamp=false);
	void computeTinputRegions(const ImageType &input, const ImageRegionMapType &regionMap, ImageType &T_input, bool do_clamp=false, bool includeZero=false);
	void computeinvT(const ImageType &input, ImageType &Tinv);
	void computeinvTRegions(const ImageType &input, const ImageRegionMapType &regionMap, ImageType &Tinv, bool includeZero=false);
	void computeinvTMultipleRegions(const ImageType &input, const ImageMultipleRegionMapType &multipleRegionMap, ImageType &Tinv, bool includeZero=false);
	static void invT(VectorType &p, const ImageType &Tinv);
	static void invTRegions(VectorType &p, const ImageType &Tinv, int regionID);
	static ImageType invT(const ImageType &input, const ImageType &Tinv);
	static ImageType invTRegions(const ImageType &input, const ImageType &Tinv, const ImageRegionMapType &regionMap);
};

}

#endif // GAUSSIAN_TRANSFER_H

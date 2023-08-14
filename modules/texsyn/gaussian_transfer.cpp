#include <vector>
#include <cmath>
#include <algorithm>
#include "gaussian_transfer.h"

namespace TexSyn {

double GaussianTransfer::Erf(double x)
{
	// Save the sign of x
	int sign = 1;
	if (x < 0)
		sign = -1;
	x = std::abs(x);

	// A&S formula 7.1.26
	double t = double(1) / (double(1) + double(0.3275911) * x);
	double y = double(1) - (((((double(1.061405429) * t + -double(1.453152027)) * t) + double(1.421413741))
		* t + -double(0.284496736)) * t + double(0.254829592)) * t * std::exp(-x * x);

	return sign * y;
}

double GaussianTransfer::ErfInv(double x)
{
	double w, p;
	w = -std::log((double(1) - x) * (double(1) + x));
	if (w < double(5))
	{
		w = w - double(2.5);
		p = double(2.81022636e-08);
		p = double(3.43273939e-07) + p * w;
		p = -double(3.5233877e-06) + p * w;
		p = -double(4.39150654e-06) + p * w;
		p = double(0.00021858087) + p * w;
		p = -double(0.00125372503) + p * w;
		p = -double(0.00417768164) + p * w;
		p = double(0.246640727) + p * w;
		p = double(1.50140941) + p * w;
	}
	else
	{
		w = std::sqrt(w) - double(3);
		p = -double(0.000200214257);
		p = double(0.000100950558) + p * w;
		p = double(0.00134934322) + p * w;
		p = -double(0.00367342844) + p * w;
		p = double(0.00573950773) + p * w;
		p = -double(0.0076224613) + p * w;
		p = double(0.00943887047) + p * w;
		p = double(1.00167406) + p * w;
		p = double(2.83297682) + p * w;
	}
	return p * x;
}

double GaussianTransfer::CDF(double x, double mu, double sigma)
{
	double U = double(0.5) * (double(1) + GaussianTransfer::Erf((x-mu)/(sigma* sqrt(double(2)))));
	return U;
}

double GaussianTransfer::invCDF(double U, double mu, double sigma)
{
	double x = sigma*sqrt(double(2)) * GaussianTransfer::ErfInv(double(2)*U-double(1)) + mu;
	return x;
}

void GaussianTransfer::toMultipleRegions(ImageMultipleRegionMapType &imageMultipleRegions, const ImageRegionMapType &imageRegion)
{
	imageMultipleRegions.init(imageRegion.get_width(), imageRegion.get_height(), true);
	imageMultipleRegions.for_all_pixels([&] (ImageMultipleRegionMapType::DataType &pix, int x, int y)
	{
		int region = imageRegion.get_pixel(x, y);
		pix |= region;
	});
}

GaussianTransfer::GaussianTransfer() :
	m_mean(0.5),
	m_variance(1.0/10.0)
{}

void GaussianTransfer::setMean(DataType mean)
{
	m_mean = mean;
}

void GaussianTransfer::setVariance(DataType variance)
{
	m_variance = variance;
}

void GaussianTransfer::computeTinput(const ImageType &input, ImageType &T_input, bool do_clamp)
{
	computeTinputRegions(input, ImageRegionMapType(), T_input, do_clamp);
}

void GaussianTransfer::computeTinputRegions(const ImageType &input, const ImageRegionMapType &regionMap, ImageType &T_input, bool do_clamp, bool includeZero)
{
	ERR_FAIL_COND_MSG(!input.is_initialized(), "input must be initialized.");
	ERR_FAIL_COND_MSG(!T_input.is_initialized(), "T_input must be initialized.");
	ERR_FAIL_COND_MSG(input.get_width() != T_input.get_width() && input.get_height() != T_input.get_height(), 
	"input and T_input must have the same size (T_input is meant to be a Gaussianized version of input).");
	
	//Find the number of different ids
	int idCount = 0;
	if(regionMap.is_initialized())
	{
		regionMap.for_all_pixels([&] (const ImageRegionMapType::DataType &pix)
		{
			idCount = std::max(idCount, pix);
		});
		++idCount;
	}
	else
	{
		idCount += 1;
	}
	
	for(int id=0; id<idCount; ++id)
	{
		for(unsigned int d=0; d<input.get_nbDimensions(); ++d)
		{
			// Sort pixels of example image
			std::vector<PixelSortStruct> sortedInputValues;
			sortedInputValues.resize(input.get_width() * input.get_height());
		
			int i = 0;
			input.get_image(d).for_all_pixels([&] (const DataType &p, int x, int y)
			{
				ImageRegionMapType::DataType regionPix = regionMap.is_initialized() ? regionMap.get_pixel(x, y) : 0;
				if(regionPix == id || (includeZero && regionPix == 0))
				{
					sortedInputValues[i].x = x;
					sortedInputValues[i].y = y;
					sortedInputValues[i].value = p;
					++i;
				}
			});
			sortedInputValues.resize(i);

			std::sort(sortedInputValues.begin(), sortedInputValues.end());

			// Assign Gaussian value to each pixel
			for (unsigned int i = 0; i < sortedInputValues.size() ; i++)
			{
				// Pixel coordinates
				int x = sortedInputValues[i].x;
				int y = sortedInputValues[i].y;
				// Input quantile (given by its order in the sorting)
				double U = (i + 0.5) / (sortedInputValues.size());
				// Gaussian quantile
				double G = invCDF(U, m_mean, m_variance);
				if(do_clamp)
					G = std::min(std::max(G,double(0)), double(1));
				// Store
				T_input.set_pixel(x, y, d, G);
			}
		}
	}
}

void GaussianTransfer::computeinvT(const ImageType& input, ImageType & Tinv)
{
	computeinvTRegions(input, ImageRegionMapType(), Tinv, false);
}

void GaussianTransfer::computeinvTRegions(const ImageType &input, const ImageRegionMapType &regionMap, ImageType &Tinv, bool includeZero)
{
	ERR_FAIL_COND_MSG(!input.is_initialized(), "input must be initialized.");
	ERR_FAIL_COND_MSG(!Tinv.is_initialized(), "Tinv must be initialized.");
	
	//Find the number of different ids and yes I like wasting time by re-computing it every function for the sake of readability
	int idCount = 0;
	if(regionMap.is_initialized())
	{
		regionMap.for_all_pixels([&] (const ImageRegionMapType::DataType &pix)
		{
			idCount = std::max(idCount, pix);
		});
		++idCount;
	}
	else
	{
		idCount += 1;
	}

	ERR_FAIL_COND_MSG(idCount != Tinv.get_height(), "number of ids does not match the size of Tinv.");
	
	for(int id=0; id<idCount; ++id)
	{
		for(unsigned int d=0; d<input.get_nbDimensions(); ++d)
		{
			// Sort pixels of example image
			std::vector<DataType> sortedInputValues;
			sortedInputValues.resize(input.get_width() * input.get_height());
			int i=0;
			input.get_image(d).for_all_pixels([&] (const DataType &p, int x, int y)
			{
				ImageRegionMapType::DataType regionPix = regionMap.is_initialized() ? regionMap.get_pixel(x, y) : 0;
				if(regionPix == id || (includeZero && regionPix == 0))
				{
					sortedInputValues[i] = p;
					++i;
				}
			});
			sortedInputValues.resize(i);

			std::sort(sortedInputValues.begin(), sortedInputValues.end());

			// Generate Tinv look-up table
			for (int i = 0; i < Tinv.get_width(); i++)
			{
				// Gaussian value in [0, 1]
				double G = (i + 0.5) / (Tinv.get_width());
				// Quantile value
				double U = CDF(G, m_mean, m_variance);
				// Find quantile in sorted pixel values
				int index = int(std::floor(U * sortedInputValues.size()));
				// Get input value
				DataType value = sortedInputValues[index];
				// Store in LUT
				Tinv.set_pixel(i, id, d, value);
			}
		}
	}
}

void GaussianTransfer::computeinvTMultipleRegions(const ImageType &input, const ImageMultipleRegionMapType &multipleRegionMap, ImageType &Tinv, bool includeZero)
{
	ERR_FAIL_COND_MSG(!input.is_initialized(), "input must be initialized.");
	ERR_FAIL_COND_MSG(!Tinv.is_initialized(), "Tinv must be initialized.");
	
	//Find the number of different ids and yes I like wasting time by re-computing it every function for the sake of readability
	int idCount = 0;
	if(multipleRegionMap.is_initialized())
	{
		multipleRegionMap.for_all_pixels([&] (const ImageMultipleRegionMapType::DataType &pix, int x, int y)
		{
			idCount = std::max(idCount, pix.log2());
		});
		++idCount;
	}
	else
	{
		idCount += 1;
	}

	ERR_FAIL_COND_MSG(idCount != Tinv.get_height(), "number of ids does not match the size of Tinv.");
	
	for(int id=0; id<idCount; ++id)
	{
		for(unsigned int d=0; d<input.get_nbDimensions(); ++d)
		{
			// Sort pixels of example image
			std::vector<DataType> sortedInputValues;
			sortedInputValues.resize(input.get_width() * input.get_height());
			int i=0;
			input.get_image(d).for_all_pixels([&] (const DataType &p, int x, int y)
			{
 				ImageMultipleRegionMapType::DataType regionPix = multipleRegionMap.get_pixel(x, y);
				if((regionPix & id).toBool() || (includeZero && (regionPix.hi == 0 && regionPix.lo == 1)))
				{
					sortedInputValues[i] = p;
					++i;
				}
			});
			sortedInputValues.resize(i);

			std::sort(sortedInputValues.begin(), sortedInputValues.end());

			// Generate Tinv look-up table
			for (int i = 0; i < Tinv.get_width(); i++)
			{
				// Gaussian value in [0, 1]
				double G = (i + 0.5) / (Tinv.get_width());
				// Quantile value
				double U = CDF(G, m_mean, m_variance);
				// Find quantile in sorted pixel values
				int index = int(std::floor(U * sortedInputValues.size()));
				// Get input value
				DataType value = sortedInputValues[index];
				// Store in LUT
				Tinv.set_pixel(i, id, d, value);
			}
		}
	}
}

void GaussianTransfer::invT(VectorType &p, const ImageType &Tinv)
{
	invTRegions(p, Tinv, 0);
}

void GaussianTransfer::invTRegions(VectorType &p, const ImageType &Tinv, int regionID)
{
	ERR_FAIL_COND_MSG(p.size() == 0, "empty vector passed to invT.");
	ERR_FAIL_COND_MSG(!Tinv.is_initialized(), "Tinv must be initialized.");
	ERR_FAIL_COND_MSG(regionID >= Tinv.get_height(), "region ID out of bounds.");
	int size = Tinv.get_width() - 1;
	for(unsigned int d = 0; d < Tinv.get_nbDimensions(); ++d) 
	{
		DataType value = p[d] * size;
		int index = std::max(std::min(Tinv.get_width()-1, int(std::round(value))), 0);
		DataType tmp = Tinv.get_pixel(index, regionID, d);
		p.write[d] = tmp;
	}
}

GaussianTransfer::ImageType GaussianTransfer::invT(const ImageType &input, const ImageType &Tinv)
{
	return invTRegions(input, Tinv, ImageRegionMapType());
}

GaussianTransfer::ImageType GaussianTransfer::invTRegions(const ImageType &input, const ImageType &Tinv, const ImageRegionMapType &regionMap)
{
	ERR_FAIL_COND_V_MSG(!input.is_initialized(), ImageType(), "input must be initialized.");
	ERR_FAIL_COND_V_MSG(!Tinv.is_initialized(), ImageType(), "Tinv must be initialized.");
	ImageType output;
	output.init(input.get_width(), input.get_height(), input.get_nbDimensions());
	input.get_image(0).for_all_pixels([&] (const DataType &pix,int x,int y)
	{
		VectorType p = input.get_pixel(x, y);
		if(regionMap.is_initialized())
		{
			int regionID = regionMap.get_pixel(x, y);
			invTRegions(p, Tinv, regionID);
		}
		else
		{
			invT(p, Tinv);
		}
		output.set_pixel(x, y, p);
	});
	return output;
}

}

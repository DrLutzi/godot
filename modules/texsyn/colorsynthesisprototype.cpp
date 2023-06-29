#include "colorsynthesisprototype.h"

#define CSP_BACKGROUND_REGION 0

namespace TexSyn
{

ColorSynthesisPrototype::ColorSynthesisPrototype():
	m_FGBGMap(),
	m_strictFGBGMap(),
	m_localMeans(),
	m_foregroundRegionMap(),
	m_backgroundRegionMap(),
	m_exemplar(),
	m_covarianceMatrixForeground(),
	m_covarianceMatrixBackground(),
	m_meanForeground()
{}

void ColorSynthesisPrototype::setExemplar(const ImageType &exemplar)
{
	m_exemplar = exemplar;
}

void ColorSynthesisPrototype::setFGBGMap(const ImageScalarType &FGBGMap)
{
	m_FGBGMap = FGBGMap;
}

void ColorSynthesisPrototype::setForegroundRegionMap(const ImageRegionMapType &regionMap)
{
	m_foregroundRegionMap = regionMap;
}

void ColorSynthesisPrototype::setStrictFGBGMap(const ImageScalarType &strictFGBGMap)
{
	m_strictFGBGMap = strictFGBGMap;
}

void ColorSynthesisPrototype::computeRegionsFromStrictFGBGMap()
{
//	auto func_isForeground = [&] (int x, int y)
//	{
//		const ImageRegionMapType::DataType &strictFGBGPix = m_strictFGBGMap.get_pixel(x, y);
//		return strictFGBGPix > 0.5;
//	};

//	m_foregroundRegionMap.init(m_strictFGBGMap.get_width(), m_strictFGBGMap.get_height());
//	m_foregroundRegionMap.for_all_pixels([&] (ImageRegionMapType::DataType &pix)
//	{
//		pix = 0;
//	});
//	int id = 1;
//	m_foregroundRegionMap.for_all_pixels([&] (ImageRegionMapType::DataType &pix, int x, int y)
//	{
//		if(pix == 0 && func_isForeground(x, y))
//		{
//			seedFill(m_foregroundRegionMap, x, y, id, func_isForeground);
//			++id;
//		}
//	});
//	bool fullyFilled = false; //need to update "fullyFilled"
//	while(!fullyFilled)
//	{
//		ImageRegionMapType foregroundRegionMapCopy = m_foregroundRegionMap;
//		for(unsigned int secondID = 1; secondID<id; ++i)
//		{
//			m_foregroundRegionMap.for_all_pixels([&] (ImageRegionMapType::DataType &pix, int x, int y)
//			{
//				if(pix != 0)
//				{
//					increaseFillOnce(foregroundRegionMapCopy, x, y, [&] (int x2, int y2) -> bool
//					{
//						return m_foregroundRegionMap.pixelIsValid(x2, y2) && func_isForeground(x2, y2);
//					});
//				}
//			});
//		}
//		m_foregroundRegionMap = foregroundRegionMapCopy;
//	}
}

void ColorSynthesisPrototype::computeMeans()
{
	ERR_FAIL_COND_MSG(!m_exemplar.is_initialized(), "exemplar is not initialized.");
	m_localMeans.init(m_exemplar.get_width(), m_exemplar.get_height(), m_exemplar.get_nbDimensions());
	
	//Find the number of different ids
	int idCount = 0;
	m_foregroundRegionMap.for_all_pixels([&] (const ImageRegionMapType::DataType &pix)
	{
		idCount = std::max(idCount, pix);
	});
	++idCount;
	m_meanForeground.resize(m_exemplar.get_nbDimensions());
	
	double globalTotalWeight = 0.0;
	//For each id, compute a local mean as an array type
	for(int id=1; id<idCount; ++id)
	{
		Eigen::VectorXd localMean;
		localMean.resize(idCount-1);
		double localTotalWeight = 0.0;
		m_foregroundRegionMap.for_all_pixels([&] (const ImageRegionMapType::DataType &pix, int x, int y)
		{
			if(pix == id)
			{
				double weight = m_FGBGMap.get_pixel(x, y);
				//TODO: This is a minor readability issue.
				//I would like to convert ImageScalar/Vector to Eigen so I can easily add vectors together
				for(unsigned int d=0; d<m_exemplar.get_nbDimensions(); ++d)
				{
					ImageType::DataType pix = m_exemplar.get_pixel(x, y, d);
					localMean(d) += pix*weight;
					m_meanForeground(d) += pix*weight;
				}
				localTotalWeight += weight;
				globalTotalWeight += weight;
			}
		});
		localMean /= localTotalWeight;
		
		//writting the mean everywhere for this id, weighted by the contribution of the foreground.
		m_foregroundRegionMap.for_all_pixels([&] (const ImageRegionMapType::DataType &pix, int x, int y)
		{
			if(pix == id)
			{
				ImageScalarType::DataType pixFGBG = m_FGBGMap.get_pixel(x, y);
				for(unsigned int d=0; d<m_exemplar.get_nbDimensions(); ++d)
				{
					m_localMeans.set_pixel(x, y, d, localMean(d) * pixFGBG);
				}
			}
		});
	}
	m_meanForeground /= globalTotalWeight;
}

void ColorSynthesisPrototype::computeForegroundCovarianceMatrix(float weightThreshold)
{
	//Add weigthed pixel values to a matrix
	Eigen::MatrixXd m;
	m.resize(m_exemplar.get_width()*m_exemplar.get_height(), m_exemplar.get_nbDimensions());
	m.setZero();
	int u = 0;
	m_FGBGMap.for_all_pixels([&] (const ImageScalarType::DataType &pix, int x, int y)
	{
		if(pix >= weightThreshold)
		{
			for(unsigned int v = 0; v<m_exemplar.get_nbDimensions(); ++v)
			{
				m(u, v) = m_exemplar.get_pixel(x, y, v);
			}
			++u;
		}
	});
	
	//Putting this into a matrix and computing it from the Eigen lib
	m.conservativeResize(u+1, m_exemplar.get_nbDimensions());
	Eigen::VectorXd mean = m.colwise().mean();
	Eigen::MatrixXd centered = m.rowwise() - mean.transpose();
	if(m.rows() == 1)
	{
		m_covarianceMatrixForeground.resize(1, 1);
		m_covarianceMatrixForeground(0, 0) = 0.0;
	}
	else
	{
		m_covarianceMatrixForeground = (centered.adjoint() * centered) / double(m.rows() - 1);
	}
}

ColorSynthesisPrototype::ImageType ColorSynthesisPrototype::createFullTexture()
{
	ERR_FAIL_COND_V_MSG(!m_exemplar.is_initialized(), ImageType(), "exemplar is not initialized.");
	ERR_FAIL_COND_V_MSG(!m_FGBGMap.is_initialized(), ImageType(), "foreground/background map is not initialized.");
	ERR_FAIL_COND_V_MSG(!m_foregroundRegionMap.is_initialized(), ImageType(), "foreground region map is not initialized.");
	ImageType result;
	result.init(m_exemplar.get_width(), m_exemplar.get_height(), m_exemplar.get_nbDimensions());
	result.for_all_images([&] (ImageType::ImageScalarType &image, unsigned int d)
	{
		image.for_all_pixels([&] (ImageType::DataType &pix, int x, int y)
		{
			ImageType::DataType exemplarPix = m_exemplar.get_pixel(x, y, d);
			ImageScalarType::DataType pixFGBG = m_FGBGMap.get_pixel(x, y);
			ImageRegionMapType::DataType region = m_foregroundRegionMap.get_pixel(x, y);
			if(region != CSP_BACKGROUND_REGION)
			{
				ImageType::DataType localMean = m_localMeans.get_pixel(x, y, d);
				Eigen::VectorXd randomMean = drawNormalMultivariateDistribution(region);
				
				pix = std::min(1.0, std::max(0.0, (exemplarPix + (randomMean(d) - localMean))*pixFGBG + exemplarPix * (1.0-pixFGBG)));
			}
			else
			{
				pix = exemplarPix;
			}
		});
	});
	return result;
}

const ColorSynthesisPrototype::ImageType &ColorSynthesisPrototype::getLocalMeans() const
{
	return m_localMeans;
}

void ColorSynthesisPrototype::seedFill(	ImageRegionMapType &regionMap, int x, int y, ImageRegionMapType::DataType value,
										const bool &func_validPixel(int, int))
{
	if(!regionMap.pixelIsValid(x, y))
		return;
	ImageRegionMapType::DataType pix = regionMap.get_pixel(x, y);
	if(pix != 0)
		return;
	if(func_validPixel(x, y))
	{
		pix = value;
		seedFill(regionMap, x+1, y, value, func_validPixel);
		seedFill(regionMap, x, y+1, value, func_validPixel);
		seedFill(regionMap, x-1, y, value, func_validPixel);
		seedFill(regionMap, x, y-1, value, func_validPixel);
	}
}

void ColorSynthesisPrototype::increaseFillOnce(	ImageRegionMapType &regionMap, int x, int y,
												const bool func_validPixel(int, int))
{
	if(!regionMap.pixelIsValid(x, y))
		return;
	ImageRegionMapType::DataType pix = regionMap.get_pixel(x, y);
	if(pix != 0)
		return;
	if(func_validPixel(x+1, y))
		regionMap.set_pixel(x+1, y, pix);
	if(func_validPixel(x, y+1))
		regionMap.set_pixel(x, y+1, pix);
	if(func_validPixel(x-1, y))
		regionMap.set_pixel(x-1, y, pix);
	if(func_validPixel(x, y-1))
		regionMap.set_pixel(x, y-1, pix);
}

void ColorSynthesisPrototype::initNormalMultivariateDistribution()
{
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(m_covarianceMatrixForeground);
	m_multivariateTransform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
}

Eigen::VectorXd ColorSynthesisPrototype::drawNormalMultivariateDistribution(uint64_t seed)
{
	ERR_FAIL_COND_V_MSG(m_multivariateTransform.size() == 0, Eigen::VectorXd(), 
	"init_normal_multivariate_distribution() must be called first.");
	if(seed != 0)
	{
		m_rand.set_seed(seed);
	}
	Eigen::VectorXd result;
	result.resize(m_meanForeground.size());
	for(int i=0; i<result.size(); ++i)
	{
		result(i) = m_rand.randfn();
	}
	result = m_meanForeground + m_multivariateTransform * result;
	for(int i=0; i<result.size(); ++i)
	{
		result(i) = std::max(0.0, std::min(1.0, result(i)));
	}
	return result;
}

}

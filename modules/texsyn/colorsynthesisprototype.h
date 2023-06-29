#ifndef COLORSYNTHESISPROTOTYPE_H
#define COLORSYNTHESISPROTOTYPE_H

#include "core/math/random_number_generator.h"
#include "image_vector.h"
#include "eigen/Eigen/Core"
#include "eigen/Eigen/Dense"

namespace TexSyn
{

class ColorSynthesisPrototype
{
public:
	using ImageScalarType = ImageScalar<double>;
	using ImageRegionMapType = ImageScalar<int>;
	using ImageType = ImageVector<double>;
	using ArrayType = ImageType::VectorType;
	
	ColorSynthesisPrototype();
	
	void setExemplar(const ImageType &exemplar);
	void setFGBGMap(const ImageScalarType &FGBGMap);
	void setForegroundRegionMap(const ImageRegionMapType &regionMap);
	void setStrictFGBGMap(const ImageScalarType &strictFGBGMap);
	void computeRegionsFromStrictFGBGMap();
	void computeMeans();
	void computeForegroundCovarianceMatrix(float weightThreshold = 0.5);
	void initNormalMultivariateDistribution();
	Eigen::VectorXd drawNormalMultivariateDistribution(uint64_t seed = 0);
	
	//debug
	ImageType createFullTexture();
	const ImageType &getLocalMeans() const;
	
private:

	void seedFill(	ImageRegionMapType &regionMap, int x, int y, ImageRegionMapType::DataType value, 
					const bool &func_validPixel(int, int));
	void increaseFillOnce(ImageRegionMapType &regionMap, int x, int y,
							const bool func_validPixel(int, int));
	
	ImageScalarType m_FGBGMap;						///<scalar map representing what is a foreground or background element
	ImageScalarType m_strictFGBGMap;				///<same as above, but binary (either foreground or backgrond)
	ImageType m_localMeans;							///<texture storing the locally stationary means
	ImageRegionMapType m_foregroundRegionMap;		///<map storing an ID for each individual region of the foreground
	ImageRegionMapType m_backgroundRegionMap;		///<map storing an ID for each individual region of the background
	ImageType m_exemplar;							///<synthesis exemplar
	Eigen::MatrixXd m_covarianceMatrixForeground;	///<covariance of the foreground, for seeding new means
	Eigen::MatrixXd m_covarianceMatrixBackground;	///<covariance of the background, for seeding new means
	Eigen::VectorXd m_meanForeground;				///<mean foreground
	RandomNumberGenerator m_rand;
	Eigen::MatrixXd m_multivariateTransform;
};

}

#endif // COLORSYNTHESISPROTOTYPE_H

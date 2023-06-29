#ifndef TEXSYN_WASSERSTEIN_H
#define TEXSYN_WASSERSTEIN_H

#include "image_vector.h"

namespace TexSyn
{

class DistanceFunction
{
public:
	using ImageScalarType = ImageScalar<double>;
	using ImageType = ImageVector<double>;

	static double l2Distance(const ImageType &im1, const ImageType &im2);
};

class Wasserstein
{
public:

	using ImageType = ImageVector<double>;

	struct TextureAndWeight
	{
		ImageType image;
		double interpolationWeight;
	};

	using VectorType = LocalVector<TextureAndWeight>;
	using DistanceFunctionType = std::function<double (const ImageType&, const ImageType&)>;

	Wasserstein();

	void addTexture(const ImageType &image, double weight=0.0);
	void setWeight(unsigned int i, double weight);
	void setDistanceFunction(const DistanceFunctionType &function);
    void setSinkhornAlgorithmParameters(double regularization, int nbIterations);

	void computeOptimalTransportPlan();

	ImageType interpolate();

private:
	DistanceFunctionType m_distanceFunction;
	VectorType m_data;
    double m_sinkhornRegularization;
    int m_sinkhornNbIterations;
};

};

#endif

#ifndef TEXSYN_STATISTICS_H
#define TEXSYN_STATISTICS_H

#include "image_vector.h"
#include <cmath>

namespace TexSyn
{

template<typename T>
class Matrix
{
	using DataType=T;
	using VectorType=Vector<DataType>;

	Matrix();
	Matrix(unsigned int rows, unsigned int cols);

	void resize(unsigned int rows, unsigned int cols);
	DataType &operator()(unsigned int row, unsigned int col);
	const DataType &operator()(unsigned int row, unsigned int col) const;

private:

	void inline toRCFromI(int i, int &r, int &c) const {c = i%m_nbRows; r=i/m_nbCols;}
	void inline toIFromRC(int &i, int r, int c) const {i = r*m_nbCols + c;}

	unsigned int m_nbRows;
	unsigned int m_nbCols;
	VectorType m_data;
};

template<typename T>
Matrix<T>::Matrix() :
	m_nbRows(0),
	m_nbCols(0),
	m_data()
{}

template<typename T>
Matrix<T>::Matrix(unsigned int rows, unsigned int cols) :
	m_nbRows(rows),
	m_nbCols(cols),
	m_data(rows*cols)
{}

template<typename T>
void Matrix<T>::resize(unsigned int rows, unsigned int cols)
{
	m_nbRows=rows;
	m_nbCols=cols;
	m_data.resize(rows*cols);
}

template<typename T>
typename Matrix<T>::DataType &Matrix<T>::operator()(unsigned int row, unsigned int col)
{
	unsigned int i;
	toIFromRC(i, row, col);
	return m_data[i];
}

template<typename T>
const typename Matrix<T>::DataType &Matrix<T>::operator()(unsigned int row, unsigned int col) const
{
	unsigned int i;
	toIFromRC(i, row, col);
	return m_data[i];
}

template<typename T>
class StatisticsScalar
{
public:
	using DataType=T;
	using VectorType=Vector<DataType>;
	using ImageScalarType=ImageScalar<DataType>;
	using ImageScalarRef=const ImageScalar<DataType>&;

	StatisticsScalar(const ImageScalarType &imRef);

	DataType get_mean();
	DataType get_variance();
	DataType get_specificFirstMoment(int order, bool centering=true, bool reduction=false);

	ImageScalarRef get_autocovariance(bool autocorrelation = false);
	ImageScalarRef get_fourierMagnitude();
	ImageScalarRef get_fourierPhase();

	static ImageScalarRef inverseFFT(const ImageScalarType &magnitude, const ImageScalarType &phase);

private:
	void computeFourierTransform();

	ImageScalarRef m_imRef;

	DataType m_mean;
	bool m_meanComputed;

	DataType m_variance;
	bool m_varianceComputed;

	ImageScalarType m_autocovariance;
	ImageScalarType m_FourierMagnitude;
	ImageScalarType m_FourierPhase;
};

template<typename T>
StatisticsScalar<T>::StatisticsScalar(const ImageScalarType &imRef) :
	m_imRef(imRef),
	m_mean(DataType(0)),
	m_meanComputed(false),
	m_variance(DataType(0)),
	m_varianceComputed(false),
	m_autocovariance(),
	m_FourierMagnitude(),
	m_FourierPhase()
{}

template<typename T>
typename StatisticsScalar<T>::DataType StatisticsScalar<T>::get_mean()
{
	if(!m_meanComputed)
	{
		double mean = 0.;
		m_imRef.for_all_pixels([&] (const DataType &pix)
		{
			mean += pix;
		});
		mean /= m_imRef.get_width()*m_imRef.get_height();
		m_mean = DataType(mean);
	}
	m_meanComputed = true;
	return m_mean;
}

template<typename T>
typename StatisticsScalar<T>::DataType StatisticsScalar<T>::get_variance()
{
	if(!m_varianceComputed)
	{
		double variance = 0.;
		DataType mean = get_mean();
		m_imRef.for_all_pixels([&] (const DataType &pix)
		{
			variance += (pix-mean) * (pix-mean);
		});
		variance /= m_imRef.get_width()*m_imRef.get_height();
		m_variance = DataType(variance);
	}
	return m_variance;
}

template<typename T>
typename StatisticsScalar<T>::DataType StatisticsScalar<T>::get_specificFirstMoment(int order, bool centering, bool reduction)
{
	DataType mean = get_mean();
	double moment = 0.;
	m_imRef.for_all_pixels([&] (const DataType &pix)
	{
		moment += centering ? pow(pix-mean, double(order)) : pow(pix, double(order));
	});
	moment /= m_imRef.get_width()*m_imRef.get_height();
	if(reduction)
		moment /= get_variance();
	return DataType(moment);
}

template<typename T>
typename StatisticsScalar<T>::ImageScalarRef StatisticsScalar<T>::get_autocovariance(bool autocorrelation)
{
	if(!m_autocovariance.is_initialized())
	{
		DataType mean = get_mean();
		m_autocovariance.init(m_imRef.get_width(), m_imRef.get_height(), true);
		m_autocovariance.for_all_pixels([&] (DataType &cov, int dx, int dy)
		{
			cov = 0;
			double ddx, ddy;
			ddx = double(dx)/(m_autocovariance.get_width()-1);
			ddy = double(dy)/(m_autocovariance.get_height()-1);
			m_imRef.for_all_pixels([&] (const DataType &pix, int x, int y)
			{
				double xTau = double(x)/(m_imRef.get_width()-1) + ddx;
				double yTau = double(y)/(m_imRef.get_height()-1) + ddy;
				xTau = xTau - floor(xTau);
				yTau = yTau - floor(yTau);
				cov += (pix - mean)*(m_imRef.get_pixelInterp(xTau, yTau) - mean);
			});
			cov /= m_imRef.get_width()*m_imRef.get_height();
			if(autocorrelation)
			{
				DataType variance = get_variance();
				cov /= variance;
			}
		});
	}
	return m_autocovariance;
}

template<typename T>
typename StatisticsScalar<T>::ImageScalarRef StatisticsScalar<T>::get_fourierMagnitude()
{
	if(!m_FourierMagnitude.is_initialized())
		computeFourierTransform();
	return ImageScalarRef(&m_FourierMagnitude);
}

template<typename T>
typename StatisticsScalar<T>::ImageScalarRef StatisticsScalar<T>::get_fourierPhase()
{
	if(!m_FourierPhase.is_initialized())
		computeFourierTransform();
	return m_FourierPhase;
}

template<typename T>
typename StatisticsScalar<T>::ImageScalarRef StatisticsScalar<T>::inverseFFT(const ImageScalarType &magnitude,
																			  const ImageScalarType &phase)
{
	DEV_CHECK(magnitude.width() == phase.width() &&
			  phase.height() == phase.height());
	constexpr double pi = 3.14159265358979323846;
	unsigned int size = magnitude.get_width();
	ImageScalarType im_marginalOutput, im_marginalImaginary, im_real, im_imaginary, im_output;
	im_real.init(size, size);
	im_imaginary.init(size, size);
	im_marginalOutput.init(size, 1);
	im_output.init(size, size);

	im_real.for_all_pixels([&] (DataType &rePix, int u, int v)
	{
		DataType &imPixel = im_imaginary.get_pixel(u, v);
		rePix = magnitude.get_pixel(u, v) * cos(phase.get_pixel(u, v));
		imPixel = magnitude.get_pixel(u, v) * sin(phase.get_pixel(u, v));
	});
	im_marginalOutput.for_all_pixels([&] (DataType &pix, int x, int y)
	{
		for(unsigned int u=0; u<size; ++u)
		{
			pix = cos(2.0*pi*x*u/size);
			im_marginalImaginary.get_pixelRef(x, y)=sin(-2.0*pi*x*u/size) / size;
		}
	});
}

template<typename T>
void StatisticsScalar<T>::computeFourierTransform()
{
	constexpr double pi = 3.14159265358979323846;
	//v It is possible to compute it on non-square images though
	DEV_CHECK(m_imRef.get_width() == m_imRef.get_height());
	ImageScalarType im_real, im_imaginary, im_marginalReal, im_marginalImaginary;
	unsigned int size = m_imRef.get_width();
	im_real.init(size, size);
	im_imaginary.init(size, size);
	m_FourierMagnitude.init(size, size);
	m_FourierPhase.init(size, size);
	im_marginalReal.init(size, 1);
	im_marginalImaginary.init(size, 1);
	im_marginalReal.parallel_for_all_pixels([&] (DataType &pix, int u, int v)
	{
		for(unsigned int x=0; x<size; ++x)
		{
			pix = cos(-2.0*pi*x*u/size) / size;
			im_marginalImaginary.get_pixelRef(u, v)=sin(-2.0*pi*x*u/size) / size;
		}
	});
	im_real.parallel_for_all_pixels([&] (DataType &rePix, int u, int v)
	{
		for(unsigned int y=0; y<size; ++y)
		{
			for(unsigned int x=0; x<size; ++x)
			{
				DataType refPixel = m_imRef.get_pixel(x, y);
				DataType marginalRePixel = im_marginalReal.get_pixel(u, 0);
				rePix = marginalRePixel * cos(-2.0*pi*y*v/size) / size;
				rePix *= refPixel;
				DataType &imPixel = im_imaginary.get_pixelRef(u, v);
				DataType marginalImPixel = im_marginalImaginary.get_pixel(u, 0);
				imPixel = marginalImPixel * sin(-2.0*pi*y*v/size) / size;
				imPixel *= refPixel;
			}
		}
	});
	m_FourierMagnitude.parallel_for_all_pixels([&] (DataType &magPix, int u, int v)
	{
		DataType rePixel = im_real.get_pixel(u, v);
		DataType imPixel = im_imaginary.get_pixel(u, v);
		magPix = sqrt(rePixel*rePixel + imPixel*imPixel);
		DataType &phasePix = m_FourierPhase.get_pixelRef(u, v);
		phasePix = atan(imPixel/rePixel);
	});
}

template<typename T>
class StatisticsVector
{
public:

	using DataType=T;
	using VectorType=LocalVector<DataType>;
	using MatrixType=Matrix<DataType>;
	using StatisticsScalarType=StatisticsScalar<DataType>;
	using ImageScalarType=typename StatisticsScalarType::ImageScalarType;
	using ImageVectorType=ImageVector<DataType>;

	StatisticsVector(const ImageVectorType &imRef);

	VectorType get_mean();
	VectorType get_variance();

	MatrixType get_covariance();

private:
	const ImageVectorType &m_imRef;

	VectorType m_mean;

	MatrixType m_covariance;
	bool m_covarianceComputed;

	ImageVectorType m_autocovariance;
	ImageVectorType m_FourierMagnitude;
	ImageVectorType m_FourierPhase;
};

template<typename T>
StatisticsVector<T>::StatisticsVector(const ImageVectorType &imRef) :
	m_imRef(imRef)
{}

template<typename T>
typename StatisticsVector<T>::VectorType StatisticsVector<T>::get_mean()
{
	if(m_mean.size() == 0)
	{
		m_mean.resize(m_imRef.get_nbDimensions());
		m_imRef.parallel_for_all_images([&] (ImageScalarType &image, int index)
		{
			StatisticsScalarType statistics(image);
			m_mean[index] = statistics.get_mean();
		});
	}
	return m_mean;
}

template<typename T>
typename StatisticsVector<T>::VectorType StatisticsVector<T>::get_variance()
{
	if(m_covariance.size() == 0)
	{
		m_covariance.resize(m_imRef.get_nbDimensions(), m_imRef.get_nbDimensions());
		m_imRef.parallel_for_all_images([&] (ImageScalarType &image, int index)
		{
			StatisticsScalarType statistics(image);
			m_covariance(index, index) = statistics.get_variance();
		});
	}
	VectorType variance(m_imRef.get_nbDimensions());
	for(unsigned int i=0; i<variance.size(); ++i)
	{
		variance[i] = m_covariance(i, i);
	}
	return variance;
}

template<typename T>
typename StatisticsVector<T>::MatrixType StatisticsVector<T>::get_covariance()
{
	if(m_covariance.size() == 0)
	{
		get_variance();
	}
	if(!m_covarianceComputed)
	{
		unsigned int matrixSize = m_imRef.get_nbDimensions();
		for(unsigned int r=0; r<matrixSize; ++r)
		{
			for(unsigned int c=0; c<matrixSize; ++r)
			{
				if(r < c)
				{
					const ImageScalarType &image1 = m_imRef[r];
					const ImageScalarType &image2 = m_imRef[c];
					DataType cov=0;
					image1.for_all_pixels([&] (const DataType &pix, int x, int y)
					{
						const DataType &pixIm2 = image2.get_pixelRef(x, y);
						cov += (pix - m_mean[r]) * (pixIm2 - m_mean[c]);
					});
					m_covariance(r, c) = cov/(image1.get_width() * image1.get_height());
				}
				if(r > c)
				{
					m_covariance(r, c) = m_covariance(c, r);
				}
			}
		}
		m_covarianceComputed=true;
	}
	return m_covariance;
}


}

#endif // TEXSYN_STATISTICS_H

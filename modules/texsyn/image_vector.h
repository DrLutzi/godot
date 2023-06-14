#ifndef TEXSYN_IMAGE_VECTOR_H
#define TEXSYN_IMAGE_VECTOR_H

#include "core/io/image.h"
#include <functional>
#include "image_scalar.h"

#ifndef TEXSYN_REMOVE_ASSERTS
#define TEXSYN_ASSERT_DIMENSIONS(dim)						DEV_ASSERT(is_initialized() && (dim)<get_nbDimensions())
#define TEXSYN_ASSERT_DIMENSIONS_AND_IN_BOUNDS(x, y, dim)	DEV_ASSERT(is_initialized() && (dim)<get_nbDimensions()\
															&& (x) < get_width() && (y)<get_width())
#define TEXSYN_ASSERT_SAME_DIMENSIONS(other)				DEV_ASSERT(is_initialized() && (other).is_initialized()\
															&& (other).get_nbDimensions() == get_nbDimensions()\
															&& (other).get_width() == get_width()\
															&& (other).get_height() == get_height())
#else
#define TEXSYN_ASSERT_DIMENSIONS(dim)
#define TEXSYN_ASSERT_DIMENSIONS_AND_IN_BOUNDS(x, y, dim)
#define TEXSYN_ASSERT_SAME_DIMENSIONS(other)
#endif

namespace TexSyn
{

static unsigned int getNbDimensionsFromFormat(Image::Format format);

template<typename T>
class ImageVector
{
public:
	using DataType=T;
	using VectorType=Vector<DataType>;
	using ImageScalarType=ImageScalar<DataType>;
	using ImageArrayType=LocalVector<ImageScalarType>;

	ImageVector();
	ImageVector(const Image &image, unsigned int nbChannels=3);
	ImageVector(const ImageVector &other);
	ImageVector(const ImageScalarType &imageScalar);

	~ImageVector();

	void init(	unsigned int width, unsigned int height,
				unsigned int nbDimensions, bool initToZero=false);
	bool is_initialized() const;

	DataType get_pixelInterp(double x, double y, int d) const;

	DataType get_pixel(int x, int y, int d) const;
	VectorType get_pixel(int x, int y) const;
	const DataType &get_pixelRef(int x, int y, int d) const;
	DataType &get_pixelRef(int x, int y, int d);

	ImageScalarType &get_image(int d);
	const ImageScalarType &get_image(int d) const;

	ImageArrayType &get_imagesScalar();
	const ImageArrayType &get_imagesScalar() const;

	int get_width() const;
	int get_height() const;
	unsigned int get_nbDimensions() const;

	void set_pixel(int x, int y, int d, DataType value);
	void set_pixel(int x, int y, const VectorType &v);
	void set_rect(const ImageVector<T> &rect, int x, int y);

	void fromImage(Ref<Image> image);
	void toImage(Ref<Image> image) const;

	void fromImageIndexed(Ref<Image> image, unsigned int startIndex); //For anything more than RGBA
	void toImageIndexed(Ref<Image> image, unsigned int startIndex);

	ImageVector<T> &operator=(const ImageVector<T> &other);

	//Operators: between images
	ImageVector<T> &operator+=(const ImageVector<T> &other);
	ImageVector<T> &operator-=(const ImageVector<T> &other);
	ImageVector<T> &operator*=(const ImageVector<T> &other);
	ImageVector<T> &operator/=(const ImageVector<T> &other);
	ImageVector<T> &operator^=(const ImageScalar<T> &kernel);

	ImageVector<T> operator+(const ImageVector<T> &other) const;
	ImageVector<T> operator-(const ImageVector<T> &other) const;
	ImageVector<T> operator*(const ImageVector<T> &other) const;
	ImageVector<T> operator/(const ImageVector<T> &other) const;
	ImageVector<T> operator^(const ImageScalar<T> &kernel) const;

	//Operators: with a scalar
	ImageVector<T> &operator+=(const DataType &s);
	ImageVector<T> &operator-=(const DataType &s);
	ImageVector<T> &operator*=(const DataType &s);
	ImageVector<T> &operator/=(const DataType &s);

	ImageVector<T> operator+(const DataType &s) const;
	ImageVector<T> operator-(const DataType &s) const;
	ImageVector<T> operator*(const DataType &s) const;
	ImageVector<T> operator/(const DataType &s) const;

	//Operators: random access
	ImageScalar<T> &operator[](unsigned int d);
	const ImageScalar<T> &operator[](unsigned int d) const;

	void for_all_images(const std::function<void (ImageScalarType &)> &f);
	void for_all_images(const std::function<void (const ImageScalarType &)> &f) const;

	void for_all_images(const std::function<void (ImageScalarType &, unsigned int)> &f);
	void for_all_images(const std::function<void (const ImageScalarType &, unsigned int)> &f) const;

	void parallel_for_all_images(const std::function<void (ImageScalarType &)> &f);
	void parallel_for_all_images(const std::function<void (const ImageScalarType &)> &f) const;

	void parallel_for_all_images(const std::function<void (ImageScalarType &, unsigned int)> &f);
	void parallel_for_all_images(const std::function<void (const ImageScalarType &, unsigned int)> &f) const;

	void parallel_for_all_pixels(const std::function<void(DataType &)>& f);
	void parallel_for_all_pixels(const std::function<void(DataType &, int, int)>& f);

	void parallel_for_all_pixels(const std::function<void(const DataType &)>& f) const;
	void parallel_for_all_pixels(const std::function<void(const DataType &, int, int)>& f) const;

	static ImageVector<T> cos(const ImageVector<T> &img);
	
	bool pixelIsValid(int x, int y) const;

private:

	struct SubImage
	{
		SubImage(ImageVector<T> &image, int startIndex, int endIndex) :
			image(image),
			startIndex(startIndex),
			endIndex(endIndex)
		{}

		ImageVector<T> &image;
		int startIndex;
		int endIndex;
	};

	struct SubImageConst
	{
		SubImageConst(const ImageVector<T> &image, int startIndex, int endIndex) :
			image(image),
			startIndex(startIndex),
			endIndex(endIndex)
		{}

		const ImageVector<T> &image;
		int startIndex;
		int endIndex;
	};

	void parallel_for_all_images(const std::function<void (SubImageConst)> &f);
	void parallel_for_all_images(const std::function<void (SubImage)> &f);
	void parallel_for_all_images(const std::function<void (SubImage, SubImageConst)> &f,
								 const ImageVector<T> &other1);
	void parallel_for_all_images(const std::function<void (SubImage, SubImageConst, SubImageConst)> &f,
								 const ImageVector<T> &other1, const ImageVector<T> &other2);

	unsigned int m_width;
	unsigned int m_height;
	ImageArrayType m_images;
};

template<typename T>
ImageVector<T>::ImageVector():
	m_width(0),
	m_height(0),
	m_images()
{}

template<typename T>
ImageVector<T>::ImageVector(const Image &image, unsigned int nbChannels):
	m_width(image.get_width()),
	m_height(image.get_height()),
	m_images()
{
	DEV_ASSERT(nbChannels<=4);
	m_images.resize(nbChannels);
	for(unsigned int i=0; i<m_images.size(); ++i)
	{
		m_images[i].for_all_pixels([&] (DataType &pix, int x, int y)
		{
			Color c = image.get_pixel(x, y);
			pix = c[i];
		});
	}
}

template<typename T>
ImageVector<T>::ImageVector(const ImageVector &other):
	m_width(other.get_width()),
	m_height(other.get_height()),
	m_images(other.m_images)
{}

template<typename T>
ImageVector<T>::ImageVector(const ImageScalarType &imageScalar) :
	m_width(imageScalar.get_width()),
	m_height(imageScalar.get_height()),
	m_images(1)
{
	m_images[0] = imageScalar;
}

template<typename T>
ImageVector<T>::~ImageVector()
{}

template<typename T>
void ImageVector<T>::init(	unsigned int width, unsigned int height,
							unsigned int nbDimensions, bool initToZero)
{
	m_width = width;
	m_height = height;
	m_images.resize(nbDimensions);
	for(unsigned int i=0; i<m_images.size(); ++i)
	{
		m_images[i].init(m_width, m_height, initToZero);
	}
}

template<typename T>
bool ImageVector<T>::is_initialized() const
{
	return m_images.size() > 0;
}

template<typename T>
typename ImageVector<T>::DataType ImageVector<T>::get_pixelInterp(double x, double y, int d) const
{
	TEXSYN_ASSERT_DIMENSIONS_AND_IN_BOUNDS(x, y, d);
	return m_images[d].get_pixelInterp(x, y);
}

template<typename T>
typename ImageVector<T>::DataType ImageVector<T>::get_pixel(int x, int y, int d) const
{
	TEXSYN_ASSERT_DIMENSIONS_AND_IN_BOUNDS(x, y, d);
	return m_images[d].get_pixel(x, y);
}

template<typename T>
typename ImageVector<T>::VectorType ImageVector<T>::get_pixel(int x, int y) const
{
	TEXSYN_ASSERT_IN_BOUNDS(x, y);
	VectorType v;
	v.resize(get_nbDimensions());
	for(unsigned int d=0; d<get_nbDimensions(); ++d)
	{
		v[d]=get_pixel(x, y, d);
	}
	return v;
}

template<typename T>
const typename ImageVector<T>::DataType &ImageVector<T>::get_pixelRef(int x, int y, int d) const
{
	TEXSYN_ASSERT_DIMENSIONS_AND_IN_BOUNDS(x, y, d);
	return m_images[d].get_pixelRef(x, y);
}

template<typename T>
typename ImageVector<T>::DataType &ImageVector<T>::get_pixelRef(int x, int y, int d)
{
	TEXSYN_ASSERT_DIMENSIONS_AND_IN_BOUNDS(x, y, d);
	return const_cast<DataType &>(get_pixelRef(x, y, d));
}

template<typename T>
typename ImageVector<T>::ImageScalarType &ImageVector<T>::get_image(int d)
{
	TEXSYN_ASSERT_DIMENSIONS(d);
	return m_images[d];
}

template<typename T>
const typename ImageVector<T>::ImageScalarType &ImageVector<T>::get_image(int d) const
{
	TEXSYN_ASSERT_DIMENSIONS(d);
	return m_images[d];
}

template<typename T>
typename ImageVector<T>::ImageArrayType &ImageVector<T>::get_imagesScalar()
{
	return m_images;
}

template<typename T>
const typename ImageVector<T>::ImageArrayType &ImageVector<T>::get_imagesScalar() const
{
	return m_images;
}

template<typename T>
int ImageVector<T>::get_width() const
{
	return m_width;
}

template<typename T>
int ImageVector<T>::get_height() const
{
	return m_height;
}

template<typename T>
unsigned int ImageVector<T>::get_nbDimensions() const
{
	return m_images.size();
}

template<typename T>
void ImageVector<T>::set_pixel(int x, int y, int d, DataType value)
{
	TEXSYN_ASSERT_DIMENSIONS_AND_IN_BOUNDS(x, y, d);
	m_images[d].set_pixel(x, y, value);
	return;
}

template<typename T>
void ImageVector<T>::set_pixel(int x, int y, const VectorType &v)
{
	TEXSYN_ASSERT_IN_BOUNDS(x, y);
	for(unsigned int i=0; i<v.size(); ++i)
	{
		set_pixel(x, y, i, v[i]);
	}
	return;
}

template <typename T>
void ImageVector<T>::set_rect(const ImageVector<T> &rect, int x, int y)
{
	TEXSYN_ASSERT_RECT_IN_BOUNDS(rect, x, y);

	for_all_images
	(
		[&](ImageScalar<T> &img, int d) { img.set_rect(rect.get_image(d), x, y); }
	);
}

template<typename T>
void ImageVector<T>::fromImage(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.ptr() == nullptr, "image must not be null, and must be locked for read access.");
	unsigned int nbDimensions=getNbDimensionsFromFormat(image->get_format());
	init(image->get_width(), image->get_height(), nbDimensions);
	for(unsigned int i=0; i<m_images.size(); ++i)
	{
		ImageScalarType &scalarImage = m_images[i];
		scalarImage.for_all_pixels([&] (DataType &pix, int x, int y)
		{
			Color c = image->get_pixel(x, y);
			pix = c[i];
		});
	}
	return;
}

template<typename T>
void ImageVector<T>::toImage(Ref<Image> image) const
{
	TEXSYN_ASSERT_INITIALIZED();
	ERR_FAIL_COND_MSG(image.ptr() == nullptr, "image must not be null, and must be locked for write access.");
	unsigned int nbDimensionsFormat=getNbDimensionsFromFormat(image.ptr()->get_format());
	unsigned int maxDimensions = MIN(nbDimensionsFormat, get_nbDimensions());
	for(unsigned int i=0; i<maxDimensions; ++i)
	{
		m_images[i].toImage(image, i);
	}
	return;
}

template<typename T>
void ImageVector<T>::fromImageIndexed(Ref<Image> image, unsigned int startIndex)
{
	ERR_FAIL_COND_MSG(image.ptr() == nullptr, "image must not be null, and must be locked for read access.");
	unsigned int nbDimensions=getNbDimensionsFromFormat(image->get_format());
	unsigned int endIndex = startIndex+nbDimensions;
	ERR_FAIL_COND_MSG(get_nbDimensions() < endIndex,
					  "image does not fit in this ImageVector from provided startIndex (try increasing its number of dimensions and check image format?).");
	ERR_FAIL_COND_MSG(image->get_width() != get_width() || image->get_height() != get_height(),
					  "size of image is different from size of this ImageVector.");
	for(unsigned int i=startIndex; i<endIndex; ++i)
	{
		ImageScalarType &scalarImage = m_images[i];
		scalarImage.parallel_for_all_pixels([&] (DataType &pix, int x, int y)
		{
			Color c = image->get_pixel(x, y);
			pix = c[i-startIndex];
		});
	}
	return;
}

template<typename T>
void ImageVector<T>::toImageIndexed(Ref<Image> image, unsigned int startIndex)
{
	TEXSYN_ASSERT_INITIALIZED();
	ERR_FAIL_COND_MSG(image.ptr() == nullptr, "image must not be null, and must be locked for write access.");
	unsigned int nbDimensionsFormat=getNbDimensionsFromFormat(image.ptr()->get_format());
	unsigned int endIndex = startIndex+nbDimensionsFormat;
	ERR_FAIL_COND_MSG(get_nbDimensions() < endIndex,
					  "image does not fit in this ImageVector from provided startIndex (try increasing its number of dimensions and check image format?).");
	ERR_FAIL_COND_MSG(image->get_width() != get_width() || image->get_height() != get_height(),
					  "size of image is different from size of this ImageVector.");
	for(unsigned int i=startIndex; i<endIndex; ++i)
	{
		m_images[i].toImage(image, i-startIndex);
	}
	return;
}

template<typename T>
ImageVector<T> &ImageVector<T>::operator=(const ImageVector<T> &other)
{
	init(other.get_width(), other.get_height(), 0);
	m_images = other.m_images;
	return *this;
}

template<typename T>
ImageVector<T> &ImageVector<T>::operator+=(const ImageVector<T> &other)
{
	auto for_image_range = [](SubImage subImage, SubImageConst subImageOther)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] += subImageOther.image[i];
		}
	};

	parallel_for_all_images(for_image_range, other);
	return *this;
}

template<typename T>
ImageVector<T> &ImageVector<T>::operator-=(const ImageVector<T> &other)
{
	auto for_image_range = [](SubImage subImage, SubImageConst subImageOther)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] -= subImageOther.image[i];
		}
	};

	parallel_for_all_images(for_image_range, other);
	return *this;
}

template<typename T>
ImageVector<T> &ImageVector<T>::operator*=(const ImageVector<T> &other)
{
	auto for_image_range = [](SubImage subImage, SubImageConst subImageOther)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] *= subImageOther.image[i];
		}
	};

	parallel_for_all_images(for_image_range, other);
	return *this;
}

template<typename T>
ImageVector<T> &ImageVector<T>::operator/=(const ImageVector<T> &other)
{
	auto for_image_range = [](SubImage subImage, SubImageConst subImageOther)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] /= subImageOther.image[i];
		}
	};

	parallel_for_all_images(for_image_range, other);
	return *this;
}

template<typename T>
ImageVector<T> &ImageVector<T>::operator^=(const ImageScalar<T> &kernel)
{
	auto for_image_range = [kernel](SubImage subImage)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] ^= kernel;
		}
	};

	parallel_for_all_images(for_image_range);
	return *this;
}

template<typename T>
ImageVector<T> ImageVector<T>::operator+(const ImageVector<T> &other) const
{
	auto for_image_range = [](SubImage subImage, SubImageConst subImageOther1, SubImageConst subImageOther2)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] = subImageOther1.image[i] + subImageOther2.image[i];
		}
	};

	ImageVector<T> output;
	output.init(get_width(), get_height(), get_nbDimensions());
	output.parallel_for_all_images(for_image_range, *this, other);
	return output;
}

template<typename T>
ImageVector<T> ImageVector<T>::operator-(const ImageVector<T> &other) const
{
	auto for_image_range = [](SubImage subImage, SubImageConst subImageOther1, SubImageConst subImageOther2)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] = subImageOther1.image[i] - subImageOther2.image[i];
		}
	};

	ImageVector<T> output;
	output.init(get_width(), get_height(), get_nbDimensions());
	output.parallel_for_all_images(for_image_range, *this, other);
	return output;
}

template<typename T>
ImageVector<T> ImageVector<T>::operator*(const ImageVector<T> &other) const
{
	auto for_image_range = [](SubImage subImage, SubImageConst subImageOther1, SubImageConst subImageOther2)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] = subImageOther1.image[i] * subImageOther2.image[i];
		}
	};

	ImageVector<T> output;
	output.init(get_width(), get_height(), get_nbDimensions());
	output.parallel_for_all_images(for_image_range, *this, other);
	return output;
}

template<typename T>
ImageVector<T> ImageVector<T>::operator/(const ImageVector<T> &other) const
{
	auto for_image_range = [](SubImage subImage, SubImageConst subImageOther1, SubImageConst subImageOther2)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] = subImageOther1.image[i] / subImageOther2.image[i];
		}
	};

	ImageVector<T> output;
	output.init(get_width(), get_height(), get_nbDimensions());
	output.parallel_for_all_images(for_image_range, *this, other);
	return output;
}

template<typename T>
ImageVector<T> ImageVector<T>::operator^(const ImageScalar<T> &kernel) const
{
	auto for_image_range = [kernel](SubImage subImage)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] ^= kernel;
		}
	};

	ImageVector<T> output = *this;
	output.parallel_for_all_images(for_image_range);
	return output;
}

//Operators: with a scalar
template<typename T>
ImageVector<T> &ImageVector<T>::operator+=(const DataType &s)
{
	auto for_image_range = [s](SubImage subImage)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] += s;
		}
	};

	parallel_for_all_images(for_image_range);
	return *this;
}

template<typename T>
ImageVector<T> &ImageVector<T>::operator-=(const DataType &s)
{
	auto for_image_range = [s](SubImage subImage)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] -= s;
		}
	};

	parallel_for_all_images(for_image_range);
	return *this;
}

template<typename T>
ImageVector<T> &ImageVector<T>::operator*=(const DataType &s)
{
	auto for_image_range = [s](SubImage subImage)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] *= s;
		}
	};

	parallel_for_all_images(for_image_range);
	return *this;
}

template<typename T>
ImageVector<T> &ImageVector<T>::operator/=(const DataType &s)
{
	auto for_image_range = [s](SubImage subImage)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] /= s;
		}
	};

	parallel_for_all_images(for_image_range);
	return *this;
}

template<typename T>
ImageVector<T> ImageVector<T>::operator+(const DataType &s) const
{
	auto for_image_range = [s](SubImage subImage, SubImageConst subImageOther)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] = subImageOther.image[i] + s;
		}
	};

	ImageVector<T> output;
	output.init(get_width(), get_height(), get_nbDimensions());
	output.parallel_for_all_images(for_image_range, *this);
	return output;
}

template<typename T>
ImageVector<T> ImageVector<T>::operator-(const DataType &s) const
{
	auto for_image_range = [s](SubImage subImage, SubImageConst subImageOther)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] = subImageOther.image[i] - s;
		}
	};

	ImageVector<T> output;
	output.init(get_width(), get_height(), get_nbDimensions());
	output.parallel_for_all_images(for_image_range, *this);
	return output;
}

template<typename T>
ImageVector<T> ImageVector<T>::operator*(const DataType &s) const
{
	auto for_image_range = [s](SubImage subImage, SubImageConst subImageOther)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] = subImageOther.image[i] * s;
		}
	};

	ImageVector<T> output;
	output.init(get_width(), get_height(), get_nbDimensions());
	output.parallel_for_all_images(for_image_range, *this);
	return output;
}

template<typename T>
ImageVector<T> ImageVector<T>::operator/(const DataType &s) const
{
	auto for_image_range = [s](SubImage subImage, SubImageConst subImageOther)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			subImage.image[i] = subImageOther.image[i] / s;
		}
	};

	ImageVector<T> output;
	output.init(get_width(), get_height(), get_nbDimensions());
	output.parallel_for_all_images(for_image_range, *this);
	return output;
}

template<typename T>
ImageScalar<T> &ImageVector<T>::operator[](unsigned int d)
{
	TEXSYN_ASSERT_DIMENSIONS(d);
	return get_image(d);
}

template<typename T>
const ImageScalar<T> &ImageVector<T>::operator[](unsigned int d) const
{
	TEXSYN_ASSERT_DIMENSIONS(d);
	return get_image(d);
}


template<typename T>
void ImageVector<T>::for_all_images(const std::function<void (ImageScalarType &)> &f)
{
	for(unsigned int i=0; i<get_nbDimensions(); ++i)
	{
		f((*this)[i]);
	}
}

template<typename T>
void ImageVector<T>::for_all_images(const std::function<void (const ImageScalarType &)> &f) const
{
	for(unsigned int i=0; i<get_nbDimensions(); ++i)
	{
		f((*this)[i]);
	}
}

template<typename T>
void ImageVector<T>::for_all_images(const std::function<void (ImageScalarType &, unsigned int)> &f)
{
	for(unsigned int i=0; i<get_nbDimensions(); ++i)
	{
		f((*this)[i], i);
	}
}

template<typename T>
void ImageVector<T>::for_all_images(const std::function<void (const ImageScalarType &, unsigned int)> &f) const
{
	for(unsigned int i=0; i<get_nbDimensions(); ++i)
	{
		f((*this)[i], i);
	}
}

template<typename T>
void ImageVector<T>::parallel_for_all_images(const std::function<void (ImageScalarType &)> &f)
{
	TEXSYN_ASSERT_INITIALIZED();
	parallel_for_all_images([&] (SubImage subImage)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			f(subImage.image[i]);
		}
	});
}

template<typename T>
void ImageVector<T>::parallel_for_all_images(const std::function<void (const ImageScalarType &)> &f) const
{
	TEXSYN_ASSERT_INITIALIZED();
	parallel_for_all_images([&] (SubImageConst subImage)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			f(subImage.image[i]);
		}
	});
}

template<typename T>
void ImageVector<T>::parallel_for_all_images(const std::function<void (ImageScalarType &, unsigned int)> &f)
{
	TEXSYN_ASSERT_INITIALIZED();
	parallel_for_all_images([&] (SubImage subImage)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			f(subImage.image[i], i);
		}
	});
}

template<typename T>
void ImageVector<T>::parallel_for_all_images(const std::function<void (const ImageScalarType &, unsigned int)> &f) const
{
	TEXSYN_ASSERT_INITIALIZED();
	parallel_for_all_images([&] (SubImageConst subImage)
	{
		for(int i=subImage.startIndex; i<subImage.endIndex; ++i)
		{
			f(subImage.image[i], i);
		}
	});
}

template <typename T>
void ImageVector<T>::parallel_for_all_pixels(const std::function<void(DataType &)> &f)
{
	parallel_for_all_images([&f](ImageScalar<T> &image) { image.parallel_for_all_pixels(f); });
}

template <typename T>
void ImageVector<T>::parallel_for_all_pixels(const std::function<void(DataType &, int, int)> &f)
{
	parallel_for_all_images([&f](ImageScalar<T> &image) { image.parallel_for_all_pixels(f); });
}

template <typename T>
void ImageVector<T>::parallel_for_all_pixels(const std::function<void(const DataType &)> &f) const
{
	parallel_for_all_images([&f](const ImageScalar<T> &image) { image.parallel_for_all_pixels(f); });
}

template <typename T>
void ImageVector<T>::parallel_for_all_pixels(const std::function<void(const DataType &, int, int)> &f) const
{
	parallel_for_all_images([&f](const ImageScalar<T> &image) { image.parallel_for_all_pixels(f); });
}

template <typename T>
ImageVector<T> ImageVector<T>::cos(const ImageVector<T> &img)
{
	auto c = img;
	c.parallel_for_all_pixels([](DataType &pix) { pix = std::cos(pix); });

	return c;
}

template<typename T>
bool ImageVector<T>::pixelIsValid(int x, int y) const
{
	return x>=0 && x<m_width && y>=0 && y<m_height;
}

template<typename T>
void ImageVector<T>::parallel_for_all_images(const std::function<void (SubImageConst)> &f)
{
	TEXSYN_ASSERT_INITIALIZED();
	unsigned int nbThreadsHint = OS::get_singleton()->get_processor_count();
	unsigned int nbThreads = MIN(nbThreadsHint == 0 ? 8 : (nbThreadsHint), m_images.size());
	unsigned int batchSize = m_images.size() / nbThreads;
	unsigned int batchRemainder = m_images.size() % nbThreads;

	std::vector<std::thread> threads;
	threads.reserve(nbThreads);

	//Apply f on contiguous sections of the vector
	for(unsigned i=0; i<nbThreads; ++i)
	{
		int startIndex = i * batchSize;
		int endIndex = startIndex + batchSize;
		SubImageConst subImage(*this, startIndex, endIndex);
		threads.push_back(std::thread(f, subImage));
	}

	// Apply f to the rest of the elements
	int startIndex = nbThreads * batchSize;
	int endIndex = startIndex + batchRemainder;
	SubImageConst subImage(*this, startIndex, endIndex);
	f(subImage);

	//Barrier to finish all threads
	for(unsigned i=0; i<nbThreads; ++i)
	{
		threads[i].join();
	}
}

template<typename T>
void ImageVector<T>::parallel_for_all_images(const std::function<void (SubImage)> &f)
{
	TEXSYN_ASSERT_INITIALIZED();
	unsigned int nbThreadsHint = OS::get_singleton()->get_processor_count();
	unsigned int nbThreads = MIN(nbThreadsHint == 0 ? 8 : (nbThreadsHint), m_images.size());
	unsigned int batchSize = m_images.size() / nbThreads;
	unsigned int batchRemainder = m_images.size() % nbThreads;

	std::vector<std::thread> threads;
	threads.reserve(nbThreads);

	//Apply f on contiguous sections of the vector
	for(unsigned i=0; i<nbThreads; ++i)
	{
		int startIndex = i * batchSize;
		int endIndex = startIndex + batchSize;
		SubImage subImage(*this, startIndex, endIndex);
		threads.push_back(std::thread(f, subImage));
	}

	// Apply f to the rest of the elements
	int startIndex = nbThreads * batchSize;
	int endIndex = startIndex + batchRemainder;
	SubImage subImage(*this, startIndex, endIndex);
	f(subImage);

	//Barrier to finish all threads
	for(unsigned i=0; i<nbThreads; ++i)
	{
		threads[i].join();
	}
}

template<typename T>
void ImageVector<T>::parallel_for_all_images(	const std::function<void (SubImage, SubImageConst)> &f,
												const ImageVector<T> &other1)
{
	TEXSYN_ASSERT_SAME_DIMENSIONS(other1);
	unsigned int nbThreadsHint = OS::get_singleton()->get_processor_count();
	unsigned int nbThreads = MIN(nbThreadsHint == 0 ? 8 : (nbThreadsHint), m_images.size());
	unsigned int batchSize = m_images.size() / nbThreads;
	unsigned int batchRemainder = m_images.size() % nbThreads;

	std::vector<std::thread> threads;
	threads.reserve(nbThreads);

	//Apply f on contiguous sections of the vector
	for(unsigned i=0; i<nbThreads; ++i)
	{
		int startIndex = i * batchSize;
		int endIndex = startIndex + batchSize;
		SubImage subImage(*this, startIndex, endIndex);
		SubImageConst subImageOther1(other1, startIndex, endIndex);
		threads.push_back(std::thread(f, subImage, subImageOther1));
	}

	// Apply f to the rest of the elements
	int startIndex = nbThreads * batchSize;
	int endIndex = startIndex + batchRemainder;
	SubImage subImage(*this, startIndex, endIndex);
	SubImageConst subImageOther1(other1, startIndex, endIndex);
	f(subImage, subImageOther1);

	//Barrier to finish all threads
	for(unsigned i=0; i<nbThreads; ++i)
	{
		threads[i].join();
	}
}

template<typename T>
void ImageVector<T>::parallel_for_all_images(	const std::function<void (SubImage, SubImageConst, SubImageConst)> &f,
												const ImageVector<T> &other1, const ImageVector<T> &other2)
{
	TEXSYN_ASSERT_SAME_DIMENSIONS(other1);
	TEXSYN_ASSERT_SAME_DIMENSIONS(other2);
	unsigned int nbThreadsHint = OS::get_singleton()->get_processor_count();
	unsigned int nbThreads = MIN(nbThreadsHint == 0 ? 8 : (nbThreadsHint), m_images.size());
	unsigned int batchSize = m_images.size() / nbThreads;
	unsigned int batchRemainder = m_images.size() % nbThreads;

	std::vector<std::thread> threads;
	threads.reserve(nbThreads);

	//Apply f on contiguous sections of the vector
	for(unsigned i=0; i<nbThreads; ++i)
	{
		int startIndex = i * batchSize;
		int endIndex = startIndex + batchSize;
		SubImage subImage(*this, startIndex, endIndex);
		SubImageConst subImageOther1(other1, startIndex, endIndex);
		SubImageConst subImageOther2(other2, startIndex, endIndex);
		threads.push_back(std::thread(f, subImage, subImageOther1, subImageOther2));
	}

	// Apply f to the rest of the elements
	int startIndex = nbThreads * batchSize;
	int endIndex = startIndex + batchRemainder;
	SubImage subImage(*this, startIndex, endIndex);
	SubImageConst subImageOther1(other1, startIndex, endIndex);
	SubImageConst subImageOther2(other2, startIndex, endIndex);
	f(subImage, subImageOther1, subImageOther2);

	//Barrier to finish all threads
	for(unsigned i=0; i<nbThreads; ++i)
	{
		threads[i].join();
	}
}

static unsigned int getNbDimensionsFromFormat(Image::Format format)
{
	unsigned int nbDimensions;
	switch (format)
	{
		case Image::FORMAT_L8: {
			nbDimensions = 1;
		} break;
		case Image::FORMAT_LA8: {
			nbDimensions = 2;
		} break;
		case Image::FORMAT_R8: {
			nbDimensions = 1;
		} break;
		case Image::FORMAT_RG8: {
			nbDimensions = 2;
		} break;
		case Image::FORMAT_RGB8: {
			nbDimensions = 3;
		} break;
		case Image::FORMAT_RGBA8: {
			nbDimensions = 4;

		} break;
		case Image::FORMAT_RF: {
			nbDimensions = 1;
		} break;
		case Image::FORMAT_RGF: {
			nbDimensions = 2;
		} break;
		case Image::FORMAT_RGBF: {
			nbDimensions = 3;
		} break;
		case Image::FORMAT_RGBAF: {
			nbDimensions = 4;
		} break;
		case Image::FORMAT_RH: {
			nbDimensions = 1;
		} break;
		case Image::FORMAT_RGH: {
			nbDimensions = 2;
		} break;
		case Image::FORMAT_RGBH: {
			nbDimensions = 3;
		} break;
		case Image::FORMAT_RGBAH: {
			nbDimensions = 4;
		} break;
		default:
		{
			nbDimensions=0;
			//Format not supported
		}
	}
	return nbDimensions;
}

}

#endif // TEXSYN_IMAGE_VECTOR_H

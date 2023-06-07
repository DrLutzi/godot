#ifndef TEXSYN_IMAGE_SCALAR_H
#define TEXSYN_IMAGE_SCALAR_H

#include "core/io/image.h"
#include "core/templates/local_vector.h"
#include <functional>
#include "core/os/os.h"
#include <vector>
#include <thread>
#include <iostream>

#define TEXSYN_REMOVE_ASSERTS
#ifndef TEXSYN_REMOVE_ASSERTS
#define TEXSYN_ASSERT_INITIALIZED()		DEV_ASSERT(is_initialized())
#define TEXSYN_ASSERT_SAME_SIZE(other) 	DEV_ASSERT(is_initialized() && other.is_initialized()\
										&& get_width() == other.get_width()\
										&& get_height() == other.get_height())
#define TEXSYN_ASSERT_IN_BOUNDS(x, y)	DEV_ASSERT(is_initialized() && (x)<get_width() && (y)<get_height())
#define TEXSYN_ASSERT_RECT_IN_BOUNDS(rect, x, y)   DEV_ASSERT(is_initialized() && (rect).is_initialized()\
                                                  && (x)<get_width() && (x)+(rect).get_width()<get_width()\
                                                  && (y)<get_height() && (y)+(rect).get_height()<get_height())
#define TEXSYN_ASSERT_KERNEL_SUITABLE(kernel) DEV_ASSERT((kernel).is_initialized() \
											  && (kernel).get_width() == (kernel).get_height() \
											  && (kernel).get_width() % 2 == 1)
#else
#define TEXSYN_ASSERT_INITIALIZED()
#define TEXSYN_ASSERT_SAME_SIZE(other)
#define TEXSYN_ASSERT_IN_BOUNDS(x, y)
#define TEXSYN_ASSERT_RECT_IN_BOUNDS(rect, x, y)
#define TEXSYN_ASSERT_KERNEL_SUITABLE(kernel)
#endif

namespace TexSyn
{

template<typename T>
class ImageScalar
{
public:
	using DataType=T;
	using VectorType=LocalVector<DataType>;
	using PosType=Vector2i;

	ImageScalar();
	ImageScalar(const Image &image);
	ImageScalar(const ImageScalar &other);

	~ImageScalar();

	void init(unsigned int width, unsigned int height, bool initToZero=false);
	bool is_initialized() const;
	DataType get_pixel(int x, int y) const;
	DataType get_pixel(PosType pos) const;

	DataType get_pixelInterp(double x, double y) const;

	const DataType &get_pixelRef(PosType pos) const;
	DataType &get_pixelRef(PosType pos);
	const DataType &get_pixelRef(int x, int y) const;
	DataType &get_pixelRef(int x, int y);

	int get_width() const;
	int get_height() const;

	void set_pixel(int x, int y, DataType value);
    void set_rect(const ImageScalar &subImage, int size, int y);

	void fromImage(Ref<Image> image);
	void toImage(Ref<Image> image, unsigned int channel) const;

	const VectorType &get_vector() const;
	VectorType &get_vector();

	//Iterators

	struct Iterator
	{
		using self 				= Iterator;
		using iterator_category	= std::bidirectional_iterator_tag;
		using difference_type	= std::ptrdiff_t;
		using value_type		= DataType;
		using pointer			= value_type*;
		using reference			= value_type&;

		Iterator(pointer ptr) : m_ptr(ptr) {}

		reference operator*() const { return *m_ptr; }
		pointer operator->() { return m_ptr; }
		self& operator++() { m_ptr++; return *this; }
		self operator++(int) { self tmp = *this; ++(*this); return tmp; }
		self& operator--() { m_ptr--; return *this; }
		self operator--(int) { self tmp = *this; --(*this); return tmp; }

		self& operator+= (difference_type diff) {m_ptr += diff; return *this;}
		self& operator-= (difference_type diff) {m_ptr -= diff; return m_ptr;}

		self operator+ (const self& b) {return m_ptr + b.m_ptr;}
		difference_type operator- (const self& b) {return m_ptr - b.m_ptr;}
		self operator+ (difference_type diff) {return m_ptr + diff;}
		self operator- (difference_type diff) {return m_ptr - diff;}

		bool operator== (const self& b) {return m_ptr == b.m_ptr;}
		bool operator!= (const self& b) {return m_ptr != b.m_ptr;}
		bool operator>(const self& b) {return m_ptr > b.m_ptr;}
		bool operator<(const self& b) {return m_ptr < b.m_ptr;}
		bool operator>=(const self& b) {return m_ptr >= b.m_ptr;}
		bool operator<=(const self& b) {return m_ptr <= b.m_ptr;}

		private:
			pointer m_ptr;
	};

	struct ConstIterator
	{
		using self 				= ConstIterator;
		using iterator_category	= std::bidirectional_iterator_tag;
		using difference_type	= std::ptrdiff_t;
		using value_type		= DataType;
		using pointer			= const value_type*;
		using reference			= const value_type&;

		ConstIterator(pointer ptr) : m_ptr(ptr) {}

		reference operator*() const { return *m_ptr; }
		pointer operator->() { return m_ptr; }
		self& operator++() { m_ptr++; return *this; }
		self operator++(int) { self tmp = *this; ++(*this); return tmp; }
		self& operator--() { m_ptr--; return *this; }
		self operator--(int) { self tmp = *this; --(*this); return tmp; }

		self& operator+= (difference_type diff) {m_ptr += diff; return m_ptr;}
		self& operator-= (difference_type diff) {m_ptr -= diff; return m_ptr;}

		self operator+ (const self& b) {return m_ptr + b.m_ptr;}
		difference_type operator- (const self& b) {return m_ptr - b.m_ptr;}
		self operator+ (difference_type diff) {return m_ptr + diff;}
		self operator- (difference_type diff) {return m_ptr - diff;}

		bool operator== (const self& b) {return m_ptr == b.m_ptr;}
		bool operator!= (const self& b) {return m_ptr != b.m_ptr;}
		bool operator>(const self& b) {return m_ptr > b.m_ptr;}
		bool operator<(const self& b) {return m_ptr < b.m_ptr;}
		bool operator>=(const self& b) {return m_ptr >= b.m_ptr;}
		bool operator<=(const self& b) {return m_ptr <= b.m_ptr;}

		private:
			pointer m_ptr;
	};

	Iterator begin() {return m_data.ptr();}
	ConstIterator begin() const {return m_data.ptr();}
	Iterator end() {return m_data.ptr() + m_data.size();}
	ConstIterator end() const {return m_data.ptr() + m_data.size();}

	//Lambda traversal

	/// @brief iterates through the image, width first, and applies f to all of its values
	/// @param f function that takes a DataType reference/value and returns nothing
	void for_all_pixels(const std::function<void(DataType &)>& f);

	/// @brief iterates through the image, width first, and applies f to all of its values
	/// @param f function that takes a DataType reference/value and the x and y coordinates, and returns nothing
	void for_all_pixels(const std::function<void(DataType &, int, int)>& f);

	//Lambda traversal (const)
	void for_all_pixels(const std::function<void(const DataType &)>& f) const;
	void for_all_pixels(const std::function<void(const DataType &, int, int)>& f) const;

	//Lambda traversal (parallel)
	/// @brief iterates through the image in parallel and applies f to all of its values
	/// @param f function that takes a DataType reference/value and returns nothing
	void parallel_for_all_pixels(const std::function<void(DataType &)>& f);

	/// @brief iterates through the image in parallel and applies f to all of its values
	/// @param f function that takes a DataType reference/value and the x and y coordinates, and returns nothing
	void parallel_for_all_pixels(const std::function<void(DataType &, int, int)>& f);

	//Lambda traversal (parallel const)
	void parallel_for_all_pixels(const std::function<void(const DataType &)>& f) const;
	void parallel_for_all_pixels(const std::function<void(const DataType &, int, int)>& f) const;

	ImageScalar<T> &operator=(const ImageScalar<T> &other);

	//Operators: between images
	ImageScalar<T> &operator+=(const ImageScalar<T> &other);
	ImageScalar<T> &operator-=(const ImageScalar<T> &other);
	ImageScalar<T> &operator*=(const ImageScalar<T> &other);
	ImageScalar<T> &operator/=(const ImageScalar<T> &other);
    ImageScalar<T> &operator^=(const ImageScalar<T> &kernel);

	ImageScalar<T> operator+(const ImageScalar<T> &other) const;
	ImageScalar<T> operator-(const ImageScalar<T> &other) const;
	ImageScalar<T> operator*(const ImageScalar<T> &other) const;
	ImageScalar<T> operator/(const ImageScalar<T> &other) const;
    ImageScalar<T> operator^(const ImageScalar<T> &kernel) const;

	//Operators: with a scalar
	ImageScalar<T> &operator+=(const DataType &s);
	ImageScalar<T> &operator-=(const DataType &s);
	ImageScalar<T> &operator*=(const DataType &s);
	ImageScalar<T> &operator/=(const DataType &s);

	ImageScalar<T> operator+(const DataType &s) const;
	ImageScalar<T> operator-(const DataType &s) const;
	ImageScalar<T> operator*(const DataType &s) const;
	ImageScalar<T> operator/(const DataType &s) const;

private:

	void inline toXYFromI(int i, int &x, int &y) const {x = i%m_width; y=i/m_width;}
	void inline toIFromXY(int &i, int x, int y) const {i = y*m_width + x;}

	unsigned int m_width;
	unsigned int m_height;
	VectorType m_data;
};

template<typename T>
ImageScalar<T>::ImageScalar():
	m_width(0),
	m_height(0),
	m_data()
{}

template<typename T>
ImageScalar<T>::ImageScalar(const Image &image):
	m_width(image.get_width()),
	m_height(image.get_height()),
	m_data()
{
	m_data.resize(m_width * m_height);
	for_all_pixels([&] (DataType &pix, int x, int y)
	{
		Color c = image.get_pixel(x, y);
		pix = c.get_luminance();
	});
}

template<typename T>
ImageScalar<T>::ImageScalar(const ImageScalar &other):
	m_width(other.get_width()),
	m_height(other.get_height()),
	m_data(other.get_vector())
{}

template<typename T>
ImageScalar<T>::~ImageScalar()
{}

template<typename T>
void ImageScalar<T>::init(unsigned int width, unsigned int height, bool initToZero)
{
	m_width = width;
	m_height = height;
	m_data.resize(m_width * m_height);
	if(initToZero)
	{
		parallel_for_all_pixels([&] (DataType &pix)
		{
			pix = DataType(0);
		});
	}
}

template<typename T>
bool ImageScalar<T>::is_initialized() const
{
	return m_data.size() > 0;
}

template<typename T>
typename ImageScalar<T>::DataType ImageScalar<T>::get_pixel(PosType pos) const
{
	return get_pixel(pos[0], pos[1]);
}

template<typename T>
typename ImageScalar<T>::DataType ImageScalar<T>::get_pixel(int x, int y) const
{
	TEXSYN_ASSERT_IN_BOUNDS(x, y);
	int i;
	toIFromXY(i, x, y);
	return m_data[i];
}

template<typename T>
typename ImageScalar<T>::DataType ImageScalar<T>::get_pixelInterp(double x, double y) const
{
	x = CLAMP(x, 0.0, 1.0);
	y = CLAMP(y, 0.0, 1.0);
	DataType pix00, pix01, pix10, pix11;
	PosType p00, p01, p10, p11;
	double fracX, fracY;
	unsigned int width = get_width()-1, height = get_height()-1;
	double xImage = x*width, yImage = y*height;
	p00.x = Math::floor(xImage);
	p00.y = Math::floor(yImage);
	p01.x = p00.x;
	p01.y = Math::ceil(yImage);
	p10.x = Math::ceil(xImage);
	p10.y = p00.y;
	p11.x = p10.x;
	p11.y = p01.y;
	pix00 = get_pixel(p00);
	pix01 = get_pixel(p01);
	pix10 = get_pixel(p10);
	pix11 = get_pixel(p11);
	fracX = xImage - double(p00.x);
	fracY = yImage - double(p00.y);
	return		pix00 * (1.0-fracX)*(1.0-fracY)
			+	pix01 * (1.0-fracX)*fracY
			+	pix10 * fracX*(1.0-fracY)
			+	pix11 * fracX*fracY;
}

template<typename T>
const typename ImageScalar<T>::DataType &ImageScalar<T>::get_pixelRef(PosType pos) const
{
	return get_pixelRef(pos[0], pos[1]);
}

template<typename T>
typename ImageScalar<T>::DataType &ImageScalar<T>::get_pixelRef(PosType pos)
{
	return get_pixelRef(pos[0], pos[1]);
}

template<typename T>
const typename ImageScalar<T>::DataType &ImageScalar<T>::get_pixelRef(int x, int y) const
{
	TEXSYN_ASSERT_IN_BOUNDS(x, y);
	int i;
	toIFromXY(i, x, y);
	return m_data[i];
}

template<typename T>
typename ImageScalar<T>::DataType &ImageScalar<T>::get_pixelRef(int x, int y)
{
	TEXSYN_ASSERT_IN_BOUNDS(x, y);
	int i;
	toIFromXY(i, x, y);
	return m_data[i];
}

template<typename T>
int ImageScalar<T>::get_width() const
{
	return m_width;
}

template<typename T>
int ImageScalar<T>::get_height() const
{
	return m_height;
}

template<typename T>
void ImageScalar<T>::set_pixel(int x, int y, DataType value)
{
	TEXSYN_ASSERT_IN_BOUNDS(x, y);
	int i;
	toIFromXY(i, x, y);
	m_data[i] = value;
	return;
}

template <typename T>
void ImageScalar<T>::set_rect(const ImageScalar<T> &subImage, int x, int y)
{
    TEXSYN_ASSERT_RECT_IN_BOUNDS(subImage, x, y);

    const int hardware_threads = OS::get_singleton()->get_processor_count();
    const int n_threads = std::min(subImage.get_height(), hardware_threads == 0 ? 8 : hardware_threads);
    const int block_size = subImage.get_height() / n_threads;
    const int remaining_size = subImage.get_height() % n_threads;

    std::vector<std::thread> threads(n_threads);

    auto handle_line_block= [&] (const int start, const int size)
    {
        ConstIterator r_it = subImage.begin() + start*subImage.get_width();
        int i;
        toIFromXY(i, x, y+start);
        Iterator it = this->begin() + i;
        Iterator it_end = it + size*this->get_width();

        for (; it != it_end; it+=static_cast<typename Iterator::difference_type>(get_width() - subImage.get_width()))
            for (const Iterator it_line_end = it+subImage.get_width(); it != it_line_end; ++it, ++r_it)
                *it = *r_it;
    };

    int block_start = 0;
    for (int i = 0; i < n_threads; ++i)
    {
        threads[i] = std::thread(handle_line_block, block_start, block_size);
        block_start += block_size;
    }

    handle_line_block(block_start, remaining_size);

    for (int i = 0; i < n_threads; ++i)
        threads[i].join();
}

template<typename T>
void ImageScalar<T>::fromImage(Ref<Image> image)
{
	init(image->get_width(), image->get_height());
	ERR_FAIL_COND_MSG(image.ptr() == nullptr, "image must not be null, and must be locked for read access.");
	for_all_pixels([&] (DataType &pix, int x, int y)
	{
		Color c = image->get_pixel(x, y);
		pix = DataType(c.r);
	});
}

template<typename T>
void ImageScalar<T>::toImage(Ref<Image> image, unsigned int channel) const
{
	TEXSYN_ASSERT_INITIALIZED();
	for_all_pixels([&] (const DataType &pix, int x, int y)
	{
		Color c = image->get_pixel(x, y);
		c[channel] = DataType(pix);
		image->set_pixel(x, y, c);
	});
}

template<typename T>
typename ImageScalar<T>::VectorType &ImageScalar<T>::get_vector()
{
	return m_data;
}

template<typename T>
const typename ImageScalar<T>::VectorType &ImageScalar<T>::get_vector() const
{
	return m_data;
}

//Traversal functions

template<typename T>
void ImageScalar<T>::for_all_pixels(const std::function<void(DataType &)>& f)
{
	TEXSYN_ASSERT_INITIALIZED();
	for (Iterator it = begin(); it != end(); ++it)
		f(*it);
}

template<typename T>
void ImageScalar<T>::for_all_pixels(const std::function<void(DataType &, int, int)>& f)
{
	TEXSYN_ASSERT_INITIALIZED();
	int x, y;
	Iterator b = begin();
	for (Iterator it = b; it != end(); ++it)
	{
		toXYFromI(it-b, x, y);
		f(*it, x, y);
	}
}

//Lambda traversal (const)

template<typename T>
void ImageScalar<T>::for_all_pixels(const std::function<void(const DataType &)>& f) const
{
	TEXSYN_ASSERT_INITIALIZED();
	for (ConstIterator it = begin(); it != end(); ++it)
		f(*it);
}

template<typename T>
void ImageScalar<T>::for_all_pixels(const std::function<void(const DataType &, int, int)>& f) const
{
	TEXSYN_ASSERT_INITIALIZED();
	int x, y;
	ConstIterator b = begin();
	for (ConstIterator it = b; it != end(); ++it)
	{
		toXYFromI(it-b, x, y);
		f(*it, x, y);
	}
}

//Lambda traversal (parallel)
template<typename T>
void ImageScalar<T>::parallel_for_all_pixels(const std::function<void(DataType &)>& f)
{
#ifdef NO_THREADS
	for_all_pixels(f);
#else
	TEXSYN_ASSERT_INITIALIZED();
	unsigned int nbThreadsHint = OS::get_singleton()->get_processor_count();
	unsigned int nbThreads = nbThreadsHint == 0 ? 8 : (nbThreadsHint);
	unsigned int batchSize = m_data.size() / nbThreads;
	unsigned int batchRemainder = m_data.size() % nbThreads;

	//Note 1: I have not figured how to easily use the Thread class of godot,
	//I would like to avoid cluttering the class with new structs and functions though.
	//Note 2: Compile error when using LocalVector instead of std::vector, something about copied threads
	std::vector<std::thread> threads;
	threads.reserve(nbThreads);

	auto func_apply_range = [this, f] (int startIndex, int nbElements)
	{
		Iterator start = begin() + startIndex;
		Iterator end = start + nbElements;
		for (Iterator it = start; it != end; ++it)
		{
			f(*it);
		}
	};

	//Apply f on contiguous sections of the vector
	for(unsigned i=0; i<nbThreads; ++i)
	{
		int startIndex = i * batchSize;
		threads.push_back(std::thread(func_apply_range, startIndex, batchSize));
	}

	// Apply f to the rest of the elements
	int startIndex = nbThreads * batchSize;
	func_apply_range(startIndex, batchRemainder);

	//Barrier to finish all threads
	for(unsigned i=0; i<nbThreads; ++i)
	{
		threads[i].join();
	}
#endif
}

template<typename T>
void ImageScalar<T>::parallel_for_all_pixels(const std::function<void(DataType &, int, int)>& f)
{
#ifdef NO_THREADS
	for_all_pixels(f);
#else
	TEXSYN_ASSERT_INITIALIZED();
	unsigned int nbThreadsHint = OS::get_singleton()->get_processor_count();
	unsigned int nbThreads = nbThreadsHint == 0 ? 8 : (nbThreadsHint);
	unsigned int batchSize = m_data.size() / nbThreads;
	unsigned int batchRemainder = m_data.size() % nbThreads;

	//Note 1: I have not figured how to easily use the Thread class of godot,
	//I would like to avoid cluttering the class with new structs and functions though.
	//Note 2: Compile error when using LocalVector instead of std::vector, something about copied threads
	std::vector<std::thread> threads;
	threads.reserve(nbThreads);

	auto func_apply_range = [this, f] (int startIndex, int nbElements)
	{
		Iterator b = begin();
		Iterator start = b + startIndex;
		Iterator end = start + nbElements;
		int x, y;
		for (Iterator it = start; it != end; ++it)
		{
			toXYFromI(it-b, x, y);
			f(*it, x, y);
		}
	};

	//Apply f on contiguous sections of the vector
	for(unsigned i=0; i<nbThreads; ++i)
	{
		int startIndex = i * batchSize;
		threads.push_back(std::thread(func_apply_range, startIndex, batchSize));
	}

	// Apply f to the rest of the elements
	int startIndex = nbThreads * batchSize;
	func_apply_range(startIndex, batchRemainder);

	//Barrier to finish all threads
	for(unsigned i=0; i<nbThreads; ++i)
	{
		threads[i].join();
	}
#endif
}

template<typename T>
void ImageScalar<T>::parallel_for_all_pixels(const std::function<void(const DataType &)>& f) const
{
#ifdef NO_THREADS
	for_all_pixels(f);
#else
	TEXSYN_ASSERT_INITIALIZED();
	unsigned int nbThreadsHint = OS::get_singleton()->get_processor_count();
	unsigned int nbThreads = nbThreadsHint == 0 ? 8 : (nbThreadsHint);
	unsigned int batchSize = m_data.size() / nbThreads;
	unsigned int batchRemainder = m_data.size() % nbThreads;

	//Note 1: I have not figured how to easily use the Thread class of godot,
	//I would like to avoid cluttering the class with new structs and functions though.
	//Note 2: Compile error when using LocalVector instead of std::vector, something about copied threads
	std::vector<std::thread> threads;
	threads.reserve(nbThreads);

	auto func_apply_range = [this, f] (int startIndex, int nbElements)
	{
		ConstIterator start = begin() + startIndex;
		ConstIterator end = start + nbElements;
		for (ConstIterator it = start; it != end; ++it)
		{
			f(*it);
		}
	};

	//Apply f on contiguous sections of the vector
	for(unsigned i=0; i<nbThreads; ++i)
	{
		int startIndex = i * batchSize;
		threads.push_back(std::thread(func_apply_range, startIndex, batchSize));
	}

	// Apply f to the rest of the elements
	int startIndex = nbThreads * batchSize;
	func_apply_range(startIndex, batchRemainder);

	//Barrier to finish all threads
	for(unsigned i=0; i<nbThreads; ++i)
	{
		threads[i].join();
	}
#endif
}


template<typename T>
void ImageScalar<T>::parallel_for_all_pixels(const std::function<void(const DataType &, int, int)>& f) const
{
#ifdef NO_THREADS
	for_all_pixels(f);
#else
	TEXSYN_ASSERT_INITIALIZED();
	unsigned int nbThreadsHint = OS::get_singleton()->get_processor_count();
	unsigned int nbThreads = nbThreadsHint == 0 ? 8 : (nbThreadsHint);
	unsigned int batchSize = m_data.size() / nbThreads;
	unsigned int batchRemainder = m_data.size() % nbThreads;

	//Note 1: I have not figured how to easily use the Thread class of godot,
	//I would like to avoid cluttering the class with new structs and functions though.
	//Note 2: Compile error when using LocalVector instead of std::vector, something about copied threads
	std::vector<std::thread> threads;
	threads.reserve(nbThreads);

	auto func_apply_range = [this, f] (int startIndex, int nbElements)
	{
		ConstIterator b = begin();
		ConstIterator start = b + startIndex;
		ConstIterator end = start + nbElements;
		int x, y;
		for (ConstIterator it = start; it != end; ++it)
		{
			toXYFromI(it-b);
			f(*it, x, y);
		}
	};

	//Apply f on contiguous sections of the vector
	for(unsigned i=0; i<nbThreads; ++i)
	{
		int startIndex = i * batchSize;
		threads.push_back(std::thread(func_apply_range, startIndex, batchSize));
	}

	// Apply f to the rest of the elements
	int startIndex = nbThreads * batchSize;
	func_apply_range(startIndex, batchRemainder);

	//Barrier to finish all threads
	for(unsigned i=0; i<nbThreads; ++i)
	{
		threads[i].join();
	}
#endif
}

template<typename T>
ImageScalar<T> &ImageScalar<T>::operator=(const ImageScalar<T> &other)
{
	m_width = other.get_width();
	m_height = other.get_height();
	m_data = other.get_vector();
	return *this;
}

template<typename T>
ImageScalar<T> &ImageScalar<T>::operator+=(const ImageScalar<T> &other)
{
	TEXSYN_ASSERT_SAME_SIZE(other);
	Iterator localIt = begin();
	for(ConstIterator otherIt = other.begin(); localIt!=end(); ++localIt, ++otherIt)
	{
		(*localIt) += (*otherIt);
	}
	return *this;
}

template<typename T>
ImageScalar<T> &ImageScalar<T>::operator-=(const ImageScalar<T> &other)
{
	TEXSYN_ASSERT_SAME_SIZE(other);
	Iterator localIt = begin();
	for(ConstIterator otherIt = other.begin(); localIt!=end(); ++localIt, ++otherIt)
	{
		(*localIt) -= (*otherIt);
	}
	return *this;
}

template<typename T>
ImageScalar<T> &ImageScalar<T>::operator*=(const ImageScalar<T> &other)
{
	TEXSYN_ASSERT_SAME_SIZE(other);
	Iterator localIt = begin();
	for(ConstIterator otherIt = other.begin(); localIt!=end(); ++localIt, ++otherIt)
	{
		(*localIt) *= (*otherIt);
	}
	return *this;
}

template<typename T>
ImageScalar<T> &ImageScalar<T>::operator/=(const ImageScalar<T> &other)
{
	TEXSYN_ASSERT_SAME_SIZE(other);
	Iterator localIt = begin();
	for(ConstIterator otherIt = other.begin(); localIt!=end(); ++localIt, ++otherIt)
	{
		(*localIt) /= (*otherIt);
	}
	return *this;
}

template<typename T>
ImageScalar<T> &ImageScalar<T>::operator^=(const ImageScalar<T> &kernel)
{
	TEXSYN_ASSERT_KERNEL_SUITABLE(kernel);

	ImageScalar<DataType> tmp(*this);
	int k_range = kernel.get_width() / 2,
	    x_max = get_width(),
	    y_max = get_height();

    for_all_pixels
    (
        [k_range, kernel, x_max, y_max, tmp](DataType &pix, int x_pix, int y_pix)->void
        {
            DataType n_pix = static_cast<DataType>(0), clamped = static_cast<DataType>(0);
            for (int x=-k_range; x<=k_range; ++x)
            for (int y=-k_range; y<=k_range; ++y)
            {
				clamped = x_pix+x >= 0 && y_pix+y >= 0 && x_pix+x < x_max && y_pix+y < y_max ?
                          tmp.get_pixel(x_pix + x, y_pix + y) :
					      pix; // This works for gaussian blur kernel, but what about other convolutions ?
                n_pix += kernel.get_pixel(x+k_range, y+k_range) * clamped;
            }

			pix = n_pix;
        }
    );

    return *this;
}

template<typename T>
ImageScalar<T> ImageScalar<T>::operator+(const ImageScalar<T> &other) const
{
	TEXSYN_ASSERT_SAME_SIZE(other);
	ImageScalar<T> output;
	output.init(get_width(), get_height());
	ConstIterator localIt=begin(), otherIt=other.begin();
	for(Iterator it=output.begin(); it!=output.end(); ++it, ++localIt, ++otherIt)
	{
		*it = (*localIt) + (*otherIt);
	}
	return output;
}

template<typename T>
ImageScalar<T> ImageScalar<T>::operator-(const ImageScalar<T> &other) const
{
	TEXSYN_ASSERT_SAME_SIZE(other);
	ImageScalar<T> output;
	output.init(get_width(), get_height());
	ConstIterator localIt=begin(), otherIt=other.begin();
	for(Iterator it=output.begin(); it!=output.end(); ++it, ++localIt, ++otherIt)
	{
		*it = (*localIt) - (*otherIt);
	}
	return output;
}

template<typename T>
ImageScalar<T> ImageScalar<T>::operator*(const ImageScalar<T> &other) const
{
	TEXSYN_ASSERT_SAME_SIZE(other);
	ImageScalar<T> output;
	output.init(get_width(), get_height());
	ConstIterator localIt=begin(), otherIt=other.begin();
	for(Iterator it=output.begin(); it!=output.end(); ++it, ++localIt, ++otherIt)
	{
		*it = (*localIt) * (*otherIt);
	}
	return output;
}

template<typename T>
ImageScalar<T> ImageScalar<T>::operator/(const ImageScalar<T> &other) const
{
	TEXSYN_ASSERT_SAME_SIZE(other);
	ImageScalar<T> output;
	output.init(get_width(), get_height());
	ConstIterator localIt=begin(), otherIt=other.begin();
	for(Iterator it=output.begin(); it!=output.end(); ++it, ++localIt, ++otherIt)
	{
		*it = (*localIt) / (*otherIt);
	}
	return output;
}

template<typename T>
ImageScalar<T> ImageScalar<T>::operator^(const ImageScalar<T> &kernel) const
{
	TEXSYN_ASSERT_KERNEL_SUITABLE(kernel);

    ImageScalar<T> output;
    output.init(get_width(), get_height());
    int k_range = kernel.get_width() / 2,
		x_max = get_width(),
		y_max = get_height();

    output.for_all_pixels
    (
        [k_range, kernel, x_max, y_max, this](DataType &pix, int x_pix, int y_pix)->void
        {
            DataType n_pix = static_cast<DataType>(0), clamped = static_cast<DataType>(0);
            for (int x=-k_range; x<=k_range; ++x)
            for (int y=-k_range; y<=k_range; ++y)
            {
				clamped = x_pix+x >= 0 && y_pix+y >= 0 && x_pix+x < x_max && y_pix+y < y_max ?
                      get_pixel(x_pix + x, y_pix + y) :
                      static_cast<DataType>(0);
                n_pix += kernel.get_pixel(x, y) * clamped;
            }

            pix = n_pix;
        }
    );

    return output;
}

//Operators: with a scalar
template<typename T>
ImageScalar<T> &ImageScalar<T>::operator+=(const DataType &s)
{
	TEXSYN_ASSERT_INITIALIZED();
	for(Iterator it = begin(); it!=end(); ++it)
	{
		(*it) += s;
	}
	return *this;
}

template<typename T>
ImageScalar<T> &ImageScalar<T>::operator-=(const DataType &s)
{
	TEXSYN_ASSERT_INITIALIZED();
	for(Iterator it = begin(); it!=end(); ++it)
	{
		(*it) -= s;
	}
	return *this;
}

template<typename T>
ImageScalar<T> &ImageScalar<T>::operator*=(const DataType &s)
{
	TEXSYN_ASSERT_INITIALIZED();
	for(Iterator it = begin(); it!=end(); ++it)
	{
		(*it) *= s;
	}
	return *this;
}

template<typename T>
ImageScalar<T> &ImageScalar<T>::operator/=(const DataType &s)
{
	TEXSYN_ASSERT_INITIALIZED();
	for(Iterator it = begin(); it!=end(); ++it)
	{
		(*it) /= s;
	}
	return *this;
}

template<typename T>
ImageScalar<T> ImageScalar<T>::operator+(const DataType &s) const
{
	TEXSYN_ASSERT_INITIALIZED();
	ImageScalar<T> output;
	output.init(get_width(), get_height());
	ConstIterator localIt = begin();
	for(Iterator it=output.begin(); it!=output.end(); ++it, ++localIt)
	{
		*it = (*localIt) + s;
	}
	return output;
}

template<typename T>
ImageScalar<T> ImageScalar<T>::operator-(const DataType &s) const
{
	TEXSYN_ASSERT_INITIALIZED();
	ImageScalar<T> output;
	output.init(get_width(), get_height());
	ConstIterator localIt = begin();
	for(Iterator it=output.begin(); it!=output.end(); ++it, ++localIt)
	{
		*it = (*localIt) - s;
	}
	return output;
}

template<typename T>
ImageScalar<T> ImageScalar<T>::operator*(const DataType &s) const
{
	TEXSYN_ASSERT_INITIALIZED();
	ImageScalar<T> output;
	output.init(get_width(), get_height());
	ConstIterator localIt = begin();
	for(Iterator it=output.begin(); it!=output.end(); ++it, ++localIt)
	{
		*it = (*localIt) * s;
	}
	return output;
}

template<typename T>
ImageScalar<T> ImageScalar<T>::operator/(const DataType &s) const
{
	TEXSYN_ASSERT_INITIALIZED();
	ImageScalar<T> output;
	output.init(get_width(), get_height());
	ConstIterator localIt = begin();
	for(Iterator it=output.begin(); it!=output.end(); ++it, ++localIt)
	{
		*it = (*localIt) / s;
	}
	return output;
}

}

#endif // TEXSYN_IMAGE_SCALAR_H

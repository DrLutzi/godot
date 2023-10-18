#include "contentatlas.h"

ContentAtlas::ContentAtlas() :
m_contents(),
m_contributions(),
m_regionMap(),
m_origins(),
m_atlasVector(),
m_occupationMap(),
m_emplacePlan()
{}

void ContentAtlas::setContributions(const LocalVector<MipmapType> &contributions)
{
	m_contributions = contributions;
}

void ContentAtlas::setRegionMap(const ImageMultiIDType &region, unsigned int nbRegions)
{
	m_nbRegions = nbRegions;
	m_regionMap.setIDMap(region);
	m_regionMap.computeMipmap();
}

void ContentAtlas::setContentsFromAllVersions(const LocalVector<ImageVectorType> &contents)
{
	ERR_FAIL_COND_MSG(m_nbRegions == 0, "please call setRegionMap first.");
	m_contents.resize(m_nbRegions);
	for(unsigned int i=0; i<m_nbRegions; ++i)
	{
		LocalVector<MipmapType> &vec = m_contents[i];
		vec.resize(m_nbRegions);
		for(unsigned int j=0; j<m_nbRegions; ++j)
		{
			MipmapType &mipmapContent = vec[j];
			mipmapContent.setTexture(contents[i]);
			ImageVectorType &mipmapTexture = mipmapContent.mipmap(0);
			m_regionMap.mipmap(0).for_all_pixels([&] (const ImageMultiIDType::DataType &pix, int x, int y)
			{
				if(!(pix & j).toBool())
				{
					for(unsigned int d=0; d<mipmapTexture.get_nbDimensions(); ++d)
					{
						mipmapTexture.set_pixel(x, y, d, 0.);
					}
				}
			});
			mipmapContent.computeMipmap();
		}
	}
}

const LocalVector<ContentAtlas::ImageVectorType> &ContentAtlas::atlasVector() const
{
	return m_atlasVector;
}

ContentAtlas::ImageVectorType ContentAtlas::originTexture() const
{
	ImageVectorType origins;
	origins.init(m_origins.size(), m_origins[0].size(), 2, true);
	origins.get_image(0).for_all_pixels([&] (ImageVectorType::DataType &pix, int x, int y)
	{
		const PosType &origin = m_origins[x][y];
		pix = double(origin.x)/m_atlasVector[0].get_width();
	});
	origins.get_image(1).for_all_pixels([&] (ImageVectorType::DataType &pix, int x, int y)
	{
		const PosType &origin = m_origins[x][y];
		pix = double(origin.y)/m_atlasVector[0].get_height();
	});
	return origins;
}

void ContentAtlas::computeAllAtlases()
{
	m_atlasVector.resize(m_nbRegions);
	computeOriginsAndEmplacePlan();
	ERR_FAIL_COND_MSG(m_emplacePlan.size() == 0, "an emplace plan must be computed before computing atlases.");
	for(unsigned int i=0; i<m_atlasVector.size(); ++i)
	{
		ImageVectorType &atlas = m_atlasVector[i];
		computeOneAtlas(atlas, i);
	}
}

void ContentAtlas::computeOneAtlas(ImageVectorType &atlas, int contentID)
{
	for(unsigned int i=0; i<m_nbRegions; ++i)
	{
		for(int j=1; j<m_regionMap.nbMaps(); ++j)
		{
			//We want to emplace the piece of content that corresponds to the transfer of contentID in the region number i and mipmap j.
			const LocalVector<MipmapType> &mipmapContentVector = m_contents[contentID];
			const MipmapType &mipmapContent = mipmapContentVector[i];
			const ImageVectorType &content = mipmapContent.mipmap(j);
			const ImageScalarType &contribution = m_contributions[i].mipmap(j).get_image(0);
			
			AtlasIndexType atlasIndex(i, j);
			HashMap<AtlasIndexType, PosType>::ConstIterator cit1 = m_emplacePlan.find(atlasIndex);
			ERR_FAIL_COND_MSG(cit1 == m_emplacePlan.end(), "an emplace plan is missing for computing an atlas.");
			PosType origin = cit1->value;
		
			HashMap<AtlasIndexType, BoundingBox>::ConstIterator cit2 = m_boundingBoxes.find(atlasIndex);
			ERR_FAIL_COND_MSG(cit2 == m_boundingBoxes.end(), "a bounding box is missing for computing an atlas.");
			BoundingBox boundingBox = cit2->value;
			
			for(int x=0; x<boundingBox.width; ++x)
			{
				for(int y=0; y<boundingBox.height; ++y)
				{
					int xMod = (boundingBox.origin.x + x) % content.get_width();
					int yMod = (boundingBox.origin.y + y) % content.get_height();
					
					ImageScalarType::DataType contribPix = contribution.get_pixel(xMod, yMod);
					if(contribPix > 0.0)
					{
						for(unsigned int d=0; d<content.get_nbDimensions(); ++d)
						{
							atlas.set_pixel(origin.x+x, origin.y+y, d, content.get_pixel(xMod, yMod, d));
						}
					}
				}
			}
		}
	}
}

void ContentAtlas::computeOriginsAndEmplacePlan()
{
	//allocation 1
	int width = m_regionMap.mipmap(0).get_width();
	int height =  m_regionMap.mipmap(0).get_height();
	unsigned int nbDimensions = m_contents[0][0].mipmap(0).get_nbDimensions();
	m_occupationMap.init(width, height, true);
	m_origins.resize(m_nbRegions);
	for(unsigned int i=0; i<m_origins.size(); ++i)
	{
		LocalVector<PosType> &vector = m_origins[i];
		vector.resize(m_regionMap.nbMaps());
		
		ImageVectorType &atlas = m_atlasVector[i];
		atlas.init(width, height, nbDimensions);
	}
	
	for(unsigned int i=0; i<m_nbRegions; ++i)
	{
		m_origins[i][0] = PosType(0, 0);
		for(int j=1; j<m_regionMap.nbMaps(); ++j)
		{
			ReducedContent reducedContent = findReducedContentOfRegionAtResolution(i, j);
			m_reducedContentQueue.emplace(reducedContent);
		}
	}
	computeEmplacePlan();
}

ContentAtlas::ReducedContent ContentAtlas::findReducedContentOfRegionAtResolution(int region, int mipmapLevel)
{
	const ImageMultiIDType &regions = m_regionMap.mipmap(mipmapLevel);
	int xMin=regions.get_width()-1, xMax=0, yMin=regions.get_height()-1, yMax=0;
	regions.for_all_pixels([&] (const ImageMultiIDType::DataType &regionPix, int x, int y)
	{
		if((regionPix & region).toBool())
		{
			xMin = std::min(x, xMin);
			xMax = std::max(x, xMax);
			yMin = std::min(y, yMin);
			yMax = std::max(y, yMax);
		}
	});
	
	//curate x positions to take periodicity in account
	if(xMin == 0 && xMax == regions.get_width()-1)
	{
		//curation of xMin
		xMin=regions.get_width()-1; 
		bool b = true;
		while(b)
		{
			--xMin;
			b = false;
			if(xMin>0)
			{
				for(int y=yMin; y<=yMax && !b; ++y)
				{
					ImageMultiIDType::DataType regionPix = regions.get_pixel(xMin, y);
					if((regionPix & region).toBool())
					{
						b = true;
					}
				}
			}
			if(!b)
			{
				++xMin;
			}
		}
		
		//curation of xMax
		xMax=0; 
		b = true;
		while(b)
		{
			++xMax;
			b = false;
			if(xMax < regions.get_width())
			{
				for(int y=yMin; y<=yMax && !b; ++y)
				{
					ImageMultiIDType::DataType regionPix = regions.get_pixel(xMax, y);
					if((regionPix & region).toBool())
					{
						b = true;
					}
				}
			}
			if(!b)
			{
				--xMax;
			}
		}
	}
	
	//curate y positions
	if(yMin == 0 && yMax == regions.get_height()-1)
	{
		//curation of yMin
		yMin=regions.get_height()-1; 
		bool b = true;
		while(b)
		{
			--yMin;
			b = false;
			if(yMin>0)
			{
				for(int x=xMin; x<=xMax && !b; ++x)
				{
					ImageMultiIDType::DataType regionPix = regions.get_pixel(x, yMin);
					if((regionPix & region).toBool())
					{
						b = true;
					}
				}
			}
			if(!b)
			{
				++yMin;
			}
		}
		
		//curation of yMax
		yMax=0; 
		b = true;
		while(b)
		{
			++yMax;
			b = false;
			if(yMax < regions.get_height())
			{
				for(int x=xMin; x<=xMax && !b; ++x)
				{
					ImageMultiIDType::DataType regionPix = regions.get_pixel(x, yMax);
					if((regionPix & region).toBool())
					{
						b = true;
					}
				}
			}
			if(!b)
			{
				--yMax;
			}
		}
	}
	
	//Origin and size are now found, allowing us to create a hitbox
	BoundingBox bbox;
	if(xMin>xMax)
		xMax += regions.get_width();
	if(yMin>yMax)
		yMax += regions.get_height();
	bbox.origin = PosType(xMin, yMin);
	bbox.width = xMax - xMin + 1;
	bbox.height = yMax - yMin + 1;
	
	ReducedContent reducedContent;
	reducedContent.bbox = bbox;
	reducedContent.region = region;
	reducedContent.mipmapLevel = mipmapLevel;
	
	//not needed?
	reducedContent.hitbox.init(bbox.width, bbox.height);
	const ImageScalarType &contribution = m_contributions[region].mipmap(mipmapLevel).get_image(0);
	reducedContent.hitbox.for_all_pixels([&] (ImageVectorType::DataType &pix, int x, int y)
	{
		pix = contribution.get_pixel((xMin + x) % contribution.get_width(), (yMin + y) % contribution.get_height());
	});
	return reducedContent;
}

void ContentAtlas::computeEmplacePlan()
{
	m_emplacePlan.clear();
	m_boundingBoxes.clear();
	while(m_reducedContentQueue.size() > 0)
	{
		const ReducedContent &reducedContent = m_reducedContentQueue.top();
		int x = 0, y =0;
		//find a position x,y where the content could possibly be placed for each atlas.
		emplacePlan_findEmplace(reducedContent, x, y);
		ERR_FAIL_COND_MSG(y>m_occupationMap.get_height(), "Atlas does not seem to be big enough.");
		AtlasIndexType atlasIndex(reducedContent.region, reducedContent.mipmapLevel);
		PosType pos(x, y);
		m_emplacePlan.insert(atlasIndex, pos);
		m_boundingBoxes.insert(atlasIndex, reducedContent.bbox);
		m_origins[reducedContent.region][reducedContent.mipmapLevel] = pos;
		emplacePlan_doEmplace(reducedContent, x, y);
		m_reducedContentQueue.pop();
	}
}

bool ContentAtlas::emplacePlan_checkEmplace(const ReducedContent &reducedContent, int x, int y)
{
	if(x + reducedContent.hitbox.get_width() > m_occupationMap.get_width())
		return false;
	if(y + reducedContent.hitbox.get_height() > m_occupationMap.get_height())
		return false;

	int x1, y1;
	bool occupied=false;
	for(x1=0; x1<reducedContent.hitbox.get_width() && !occupied; ++x1)
	{
		for(y1=0; y1<reducedContent.hitbox.get_height() && !occupied; ++y1)
		{
			occupied = m_occupationMap.get_pixel(x+x1, y+y1)>0.0 && reducedContent.hitbox.get_pixel(x1, y1)>0.0;
		}
	}
	return !occupied;
}

void ContentAtlas::emplacePlan_findEmplace(const ReducedContent &reducedContent, int &x, int &y)
{
	//precondition: width and height of images are large enough
	while(!emplacePlan_checkEmplace(reducedContent, x, y) && (y<m_occupationMap.get_height()))
	{
		if(++y >= m_occupationMap.get_height())
		{
			y=0;
			++x;
		}
	}
}

void ContentAtlas::emplacePlan_doEmplace(const ReducedContent &reducedContent, int x, int y)
{
	reducedContent.hitbox.for_all_pixels([&] (const ImageScalarType::DataType &pix, int x1, int y1)
	{
		if(pix > 0.0)
		{
			m_occupationMap.set_pixel(x+x1, y+y1, pix);
		}
	});
}

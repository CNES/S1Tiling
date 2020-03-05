/*=========================================================================

  Program:   ORFEO Toolbox
  Language:  C++
  Date:      $Date$
  Version:   $Revision$


  Copyright (c) Centre National d'Etudes Spatiales. All rights reserved.
  See OTBCopyright.txt for details.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbMeanFilterFunctor_h
#define otbMeanFilterFunctor_h

namespace otb
{

/** \class Functor::MeanRatio
 *
 * - compute the ratio of the two pixel values
 * - compute the value of the ratio of means
 * - cast the \c double value resulting to the pixel type of the output image
 * - store the casted value into the output image.
 *

 * \ingroup Functor
 *
 * \ingroup OTBChangeDetection
 */
namespace Functor
{

template<class TInput1, class TOutput>
class MeanFilterFunctor
{
public:
  MeanFilterFunctor() {}
  virtual ~MeanFilterFunctor() {}
  inline TOutput operator ()(const TInput1& itA)
  {

    TOutput meanA=static_cast<TOutput>(0.0);
    meanA[0]=0.;

    TOutput non_zeros_pixels=static_cast<TOutput>(0.0);
    non_zeros_pixels[0]=0.;
    for (unsigned long pos = 0; pos < itA.Size(); ++pos)
      {
      if (itA.GetPixel(pos)[0]!=0.)
      {
		meanA[0] += itA.GetPixel(pos)[0];
		non_zeros_pixels[0] += 1.;
	  }
      }

    if (non_zeros_pixels[0]!=0.)
    {
		meanA[0] /= non_zeros_pixels[0];
	}
	else
	{
		meanA[0] = 0.;
	}

    //std::cout<<"meanA= "<<meanA<<", meanB= "<<meanB<<std::endl;

    TOutput ratio;
    ratio=itA.GetCenterPixel();
    if (ratio[0]!=0.)
    {
		ratio[0]=meanA[0];
	}
    //std::cout<<ratio[0]<<"\n";
    return static_cast<TOutput>(ratio);
  }
};
}
} // end namespace otb

#endif

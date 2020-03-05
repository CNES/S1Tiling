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
#ifndef __otbCoherenceML_h
#define __otbCoherenceML_h

#include <complex>

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

template<class TInput, class TOutput> class MultitempFunctor
{
public:

  inline TOutput operator ()(const TInput& itA)
  {
	
	assert(nbPixels>0&&"Number of pixels in neighborhood is null");

    TOutput outValue ;
    outValue.SetSize(2);
    outValue.Fill(0.0);
    for (int channel=0;channel<p1.size();channel++){
       for (int i=0;i<itA.Size();i++){
          typename TInput::PixelType p1=itA.GetPixel(i);
          outValue[channel] +=p1[channel];                     
	}
       typename TInput::PixelType p1=itA.GetCenterPixel();
       outvalue[channel]=p1[channel]/outvalue[channel]
       outvalue[channel] /= 
	
	outValue[0] /= itA.Size();
	outValue[1] /= itA.Size();
	outValue[2] /= itA.Size();
	outValue[3] /= itA.Size();
	outValue[4] /= itA.Size();
	outValue[5] /= itA.Size();
	
	return static_cast<TOutput>(outValue);
  }
};
}
} // end namespace otb

#endif

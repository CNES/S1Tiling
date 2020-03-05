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
#ifndef otbMultiplyVectorImageFilter_h
#define otbMultiplyVectorImageFilter_h
#include "otbMultiplyVectorImageFilterFunctor.h"

#include "itkBinaryFunctorImageFilter.h"
#include <otbImageList.h>

namespace otb
{

/** \class MeanRatioImageFilter
 * \brief Implements neighborhood-wise the computation of mean ratio.
 *
 * This filter is parametrized over the types of the two
 * input images and the type of the output image.
 *
 * Numeric conversions (castings) are done by the C++ defaults.
 *
 * The filter will walk over all the pixels in the two input images, and for
 * each one of them it will do the following:
 *
 * - compute the ratio of the two pixel values
 * - compute the value of the ratio of means
 * - cast the \c double value resulting to the pixel type of the output image
 * - store the casted value into the output image.
 *
 * The filter expect all images to have the same dimension
 * (e.g. all 2D, or all 3D, or all ND)
 *
 * \ingroup IntensityImageFilters Multithreaded
 *
 * \ingroup OTBChangeDetection
 */

template <class TInputImage1, class TInputImage2, class TOutputImage>
class ITK_EXPORT MultiplyVectorImageFilter :
  public itk::BinaryFunctorImageFilter<TInputImage1, TInputImage2, TOutputImage,
      Functor::MultiplyVectorImageFilterFunctor<typename TInputImage1::PixelType,typename TInputImage2::PixelType,typename TOutputImage::PixelType> >
{
public:
  /** Standard class typedefs. */
  typedef MultiplyVectorImageFilter Self;
  typedef itk::BinaryFunctorImageFilter<
      TInputImage1, TInputImage1,TOutputImage,
      Functor::MultiplyVectorImageFilterFunctor<typename TInputImage1::PixelType,typename TInputImage2::PixelType,typename TOutputImage::PixelType>
      >  Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Macro defining the type*/
  itkTypeMacro(MultiplyVectorImageFilter, SuperClass);

protected:
  MultiplyVectorImageFilter() {}
  ~MultiplyVectorImageFilter() ITK_OVERRIDE {}

private:
  MultiplyVectorImageFilter(const Self &); //purposely not implemented
  void operator =(const Self&); //purposely not implemented

};

} // end namespace otb

#endif

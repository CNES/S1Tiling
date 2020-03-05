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
#ifndef __otbotbCoherenceMLImageFilter_h
#define __otbotbCoherenceMLImageFilter_h

#include "otbCoherenceMLFunctor.h"
#include "otbUnaryFunctorNeighborhoodImageFilter.h"
#include "itkConstNeighborhoodIterator.h"

namespace otb
{

/** \class 
 * \brief 
 *
 * \ingroup IntensityImageFilters Multithreaded
 *
 * \ingroup 
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT CoherenceMLImageFilter :
  public UnaryFunctorNeighborhoodImageFilter<
      TInputImage, TOutputImage,
      Functor::CoherenceMLFunctor<
          typename itk::ConstNeighborhoodIterator<TInputImage>,
          typename TOutputImage::PixelType> >
{
public:
  /** Standard class typedefs. */
  typedef CoherenceMLImageFilter Self;
  typedef otb::UnaryFunctorNeighborhoodImageFilter<TInputImage, TOutputImage,
      Functor::CoherenceMLFunctor<
          typename itk::ConstNeighborhoodIterator<TInputImage>,
          typename TOutputImage::PixelType>
      >  Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Macro defining the type*/
  itkTypeMacro(CoherenceMLImageFilter, SuperClass);

protected:
  CoherenceMLImageFilter() {}
  virtual ~CoherenceMLImageFilter() {}
  virtual void GenerateOutputInformation()
	 {
		 // Call superclass implementation
		 Superclass::GenerateOutputInformation();
		 
		 this->GetOutput()->SetNumberOfComponentsPerPixel(6);
	 }
private:
  CoherenceMLImageFilter(const Self &); //purposely not implemented
  void operator =(const Self&); //purposely not implemented

};

} // end namespace otb

#endif

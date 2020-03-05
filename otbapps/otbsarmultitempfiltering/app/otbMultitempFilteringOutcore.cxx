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
#include "otbWrapperApplication.h"
#include "otbWrapperApplicationFactory.h"
#include "otbOutcoreFilter.h"
#include "otbMultiplyVectorImageFilter.h"
#include "itkAddImageFilter.h"
#include "otbImageList.h" 
#include <itkVariableLengthVector.h>
#include "otbObjectList.h"


namespace otb
{

namespace Wrapper
{

class MultitempFilteringOutcore : public Application
{
public:
  /** Standard class typedefs. */
  typedef MultitempFilteringOutcore   Self;
  typedef Application                         Superclass;
  typedef itk::SmartPointer<Self>             Pointer;
  typedef itk::SmartPointer<const Self>       ConstPointer;
  
  typedef ImageList<FloatImageType>    ImageListType;
  typedef itk::AddImageFilter<FloatVectorImageType,FloatVectorImageType,FloatVectorImageType> AddVectorImageFilterType;
  typedef typename otb::MultiplyVectorImageFilter<FloatVectorImageType,FloatVectorImageType,FloatVectorImageType> MultiplyVectorImageFilterType;
  typedef OutcoreFilter<FloatVectorImageType,FloatVectorImageType> OutcoreFilterType;

  /** Standard macro */
  itkNewMacro(Self);

  itkTypeMacro(MultitempFilteringOutcore, otb::Wrapper::Application);



private:
  void DoInit() ITK_OVERRIDE
  {
    SetName("MultitempFilteringOutcore");
    SetDescription("This application implements the Quegan speckle filter for SAR images. It computes the outcore function of the filter. It must be followed by the MultitempFilteringFilter application to compute the filtered images");

    // Documentation
    SetDocName("MultitempFilteringOutcore");
    SetDocLongDescription("This application implements the Quegan speckle filter for SAR images. It computes the outcore function of the filter. It must be followed by the MultitempFilteringFilter application to compute the filtered images" );
    
						  
    SetDocLimitations("None");
    SetDocAuthors("Thierry Koleck (CNES), Marie Ballere (CNES)");
    SetDocSeeAlso("MultitempFilteringFilter");

    AddDocTag(Tags::SAR);

    AddParameter(ParameterType_InputImageList,  "inl",   "Input images list");
    SetParameterDescription("inl", "Input image list");

    AddParameter(ParameterType_Int ,  "wr",   "Spatial averaging Window radius ");
    SetParameterDescription("wr", "Window radius");

    AddParameter(ParameterType_OutputImage, "oc",  "Outcore filename");
    SetParameterDescription("oc", "Outcore filename");
    
    AddRAMParameter();

    // Default values
  }

  void DoUpdateParameters() ITK_OVERRIDE
  {
    // Nothing to do here : all parameters are independent
  }

  void DoExecute() ITK_OVERRIDE
  {
	AddVectorImageFilterType::Pointer m_AddImageFilter; 
    MultiplyVectorImageFilterType::Pointer m_MultiplyByConstImageFilter;
    OutcoreFilterType::Pointer m_OutcoreFilter;  

    int Radius = this->GetParameterInt("wr");
    FloatVectorImageListType::Pointer inList = this->GetParameterImageList("inl");
    // On verifie que la liste en entree n'est pas vide
    std::cout << inList->Size() << "\n";
    if( inList->Size() == 0 )
      {
      itkExceptionMacro("No input Image set...");
      }
    std::vector< std::string> filelist;
    filelist=this->GetParameterStringList("inl");

    inList->GetNthElement(0)->UpdateOutputInformation();

    FilterList=otb::ObjectList<itk::ImageToImageFilter<FloatVectorImageType,FloatVectorImageType> >::New();
    
    m_AddImageFilter=AddVectorImageFilterType::New();
    m_OutcoreFilter=OutcoreFilterType::New();
    m_OutcoreFilter->SetInput(inList->GetNthElement(0));
    m_OutcoreFilter->SetRadius(Radius);
    m_OutcoreFilter->UpdateOutputInformation();
    FilterList->PushBack(m_OutcoreFilter);
    FilterList->PushBack(m_OutcoreFilter);

    for( unsigned int i=1; i<inList->Size(); i++ ) {
       m_OutcoreFilter=OutcoreFilterType::New();
       m_OutcoreFilter->SetInput(inList->GetNthElement(i));
       m_OutcoreFilter->SetRadius(Radius);
       m_OutcoreFilter->UpdateOutputInformation();
       m_AddImageFilter=AddVectorImageFilterType::New();
       m_AddImageFilter->SetInput1(m_OutcoreFilter->GetOutput());
       m_AddImageFilter->SetInput2(FilterList->GetNthElement(2*i-1)->GetOutput());
       m_AddImageFilter->UpdateOutputInformation();

       FilterList->PushBack(m_OutcoreFilter);
       FilterList->PushBack(m_AddImageFilter);
    }

    // Calcul des images de sortie
    SetParameterOutputImage("oc", FilterList->Back()->GetOutput());

  }
  otb::ObjectList<itk::ImageToImageFilter<FloatVectorImageType,FloatVectorImageType> >::Pointer FilterList;


  
}; 

} //end namespace Wrapper
} //end namespace otb

OTB_APPLICATION_EXPORT(otb::Wrapper::MultitempFilteringOutcore)

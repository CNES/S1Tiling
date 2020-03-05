To install this module in your OTB, create a file ```SARMultiTempFiltering.remote.cmake```
in your otb_install_dir/Modules/Remote directory. 

Then copy the following text in the file:

```
otb_fetch_module(SARMultiTempFiltering
"
A more detailed description can be found on the project website:
http://tully.ups-tlse.fr/koleckt/otbsarmultitempfiltering.git
"
  GIT_REPOSITORY http://tully.ups-tlse.fr/koleckt/otbsarmultitempfiltering.git
  # Commit on master branch 
  GIT_TAG master
)
```



OFILES = base/imgproc.o
         





ROOTDIR = .
LIB = $(ROOTDIR)/lib/libimgproc.a 


GLLDFLAGS     = -lglut -lGL -lm -lGLU


INSTALL_TOOL =


CXX = g++ -Wall -g -O2 -fPIC $(DEFINES) -fopenmp -std=c++11
OIIO_LIB_LOCATION = $(HOME)/projects/3rdparty/3rdbuild/lib
OIIO_INCLUDE_LOCATION = $(HOME)/projects/3rdparty/3rdbuild/include
OIIOLIB = -L${OIIO_LIB_LOCATION} -lOpenImageIO


INCLUDES =  -I $(OIIO_INCLUDE_LOCATION) -I ./include -I /usr/local/include/ -I/usr/include/ -I /opt/local/include 




.C.o:
	$(CXX) -c $(INCLUDES) $< -o $@

base: $(OFILES)
	ar rv $(LIB) $?


clean:
	rm -rf base/*.o *~  core $(LIB) img_paint 

paint:	base/img_paint.C	$(OFILES)
	$(CXX) base/img_paint.C $(INCLUDES) $(GLLDFLAGS) -L ./lib -limgproc $(OIIOLIB) -o img_paint







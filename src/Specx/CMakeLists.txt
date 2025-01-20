
  #project(SPECX CXX)


  #message("FEELPP DIR" ${FEELPP_DIR})


  include(FetchContent)

  #option( FEELPP_ENABLE_SPECX "Enable specx" ON )

#  if ( FEELPP_ENABLE_SPECX )
#	message(STATUS "Configuring specx")
#  endif()


# Move on my path

# Test if Specx works

option( FEELPP_ENABLE_SPECX "Enable specx" ON )

  if(TRUE)
	message(STATUS "------------------------------------------------------------")
	message(STATUS "Configuring specx")
  
	# set(FETCHCONTENT_SOURCE_DIR_SPECX ${CMAKE_CURRENT_SOURCE_DIR}/SPECX)

	set(SPECX_JUST_LIB ON)
	set(SPECX_DO_INSTALL ON)
	set(SPECX_COMPILE_WITH_MPI OFF)
	set(FETCHCONTENT_SOURCE_DIR_SPECX ${CMAKE_CURRENT_SOURCE_DIR}/specx)

	MESSAGE("LINK ======******=>" ${CMAKE_CURRENT_SOURCE_DIR}/SPECX)


	FetchContent_Declare(SPECX GIT_REPOSITORY https://github.com/feelpp/specx.git GIT_TAG e092e061ec17afd42374041cccfa6f44b939b19c)
  
	set(SPECX_INSTALL_CMAKE_DIR  "${CMAKE_INSTALL_DATADIR}/feel/cmake/modules")
	message(STATUS "[feelpp-specx] cmake dir: ${CMAKE_INSTALL_DATADIR}/feel/cmake/modules")
	FetchContent_MakeAvailable(specx)
	add_library(feelpp_specx INTERFACE)
  
	# set_target_properties(specx PROPERTIES OUTPUT_NAME "feelpp_specx")
	target_compile_definitions(feelpp_contrib INTERFACE FEELPP_HAS_SPECX)
	target_link_libraries(feelpp_specx INTERFACE specx )
	add_library(Feelpp::feelpp_specx ALIAS feelpp_specx)
	target_include_directories(feelpp_specx INTERFACE
	  $<BUILD_INTERFACE:${specx_BINARY_DIR}/>
	  $<BUILD_INTERFACE:${specx_BINARY_DIR}/Src>
	  $<BUILD_INTERFACE:${specx_SOURCE_DIR}/>
	  $<BUILD_INTERFACE:${specx_SOURCE_DIR}/Src>
	  $<INSTALL_INTERFACE:include/feelpp>
	  $<INSTALL_INTERFACE:include/feelpp/specx>)
	install(TARGETS feelpp_specx specx EXPORT feelpp-contrib-export-targets
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
		ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
		INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
		)
	#target_link_libraries( feelpp_contrib INTERFACE feelpp_specx )
	#set(FEELPP_ENABLED_OPTIONS_CONTRIB "${FEELPP_ENABLED_OPTIONS_CONTRIB} specx")
	
  endif()



  
  if(TRUE)

	include_directories(SYSTEM ${MPI_INCLUDES_PATH})

	set(SPECX_SOURCE_DIR ${specx_SOURCE_DIR} )
	set(SPECX_BINARY_DIR ${specx_BINARY_DIR} )

	

	#file(GLOB_RECURSE MyFiles
	#	./*.cpp
	#)

	file(GLOB MyFiles
		"*.cpp"
	)




		# Adding the project sources dir as an include dir
	INCLUDE_DIRECTORIES(
			${SPECX_BINARY_DIR}/Src    
			${SPECX_SOURCE_DIR}/Src   
			${MPI_CXX_INCLUDE_DIRS} 
			#${OpenMP_CXX_INCLUDE_DIRS} 
			#${CMAKE_CURRENT_SOURCE_DIR}/specx/Src
			#${CMAKE_CURRENT_SOURCE_DIR}/specx/Src/Data
	)

	LINK_DIRECTORIES(
		${SPECX_SOURCE_DIR}/Src   
		${SPECX_SOURCE_DIR}/Src/Data 
		${MPI_CXX_INCLUDE_DIRS} 
	)


	MESSAGE("LINK SOURCE_DIR 1======******=>" ${specx_SOURCE_DIR})
	MESSAGE("LINK BINARY_DIR 2 ======******=>" ${specx_BINARY_DIR})
	MESSAGE("LINK SPECX LIBRARIES ======******=>" ${SPECX_LIBRARIES})

	MESSAGE("LINK BINARY_DIR 2 ======******=>" ${SPECX_BINARY_DIR})
	MESSAGE("LINK SOURCE_DIR 2 ======******=>" ${SPECX_SOURCE_DIR})

	message("LINK SOURCE_DIR 3 ======******=>" ${CMAKE_INSTALL_DATADIR}"/feel/cmake/modules")


	foreach(exec ${MyFiles}) 
			#if(${SPECX_COMPILE_WITH_MPI} OR NOT ${exec} MATCHES "-mpi\.")
			#get_filename_component( execname ${exec} NAME_WE)
			get_filename_component(execname ${exec} NAME_WLE)
			get_filename_component(dirname ${exec} DIRECTORY)

				#if($ENV{VERBOSE})
				#    MESSAGE(STATUS "Benchmark Demo -- Add ${execname}")
				#endif()

				#if(SPECX_COMPILE_WITH_CUDA)
				#    set_source_files_properties(${exec} PROPERTIES LANGUAGE CUDA)
				#    set(SPECX_LIBRARIES "${SPECX_LIBRARIES};cublas;cusolver")
				#endif()
				

				message("${exec}.cpp")
				add_executable(${execname} ${exec})
				
				#set_target_properties(${execname} PROPERTIES COMPILE_FLAGS "-llapack -lblas")

				target_link_libraries(
					${execname}
					#Feelpp::feelpp_specx 
					feelpp_specx
					#feelpp_specx
					#specx
					#${SPECX_LIBRARIES}
					#${BLAS_LIBRARIES}
					#${LAPACK_LIBRARIES}
					#Feelpp::feelpp
					#${MPI_CXX_LIBRARIES}
					#${OpenMP_CXX_LIBRARIES}
					#OpenMP::OpenMP_CXX
					#feelpp_specx
				)



				#target_include_directories(
				#	${execname}
					#PUBLIC Feelpp::feelpp-contrib-export-targets
				#	PUBLIC Feelpp::feelpp_specx 
				#)
				

				#feelpp_add_application(${exec} SRCS ${execname} LINK_LIBRARIES Feelpp::feelpp_specx )

			#endif()
		endforeach(exec)
endif()


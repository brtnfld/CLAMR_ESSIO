/*
 *  Copyright (c) 2014, Los Alamos National Security, LLC.
 *  All rights Reserved.
 *
 *  Copyright 2011-2012. Los Alamos National Security, LLC. This software was produced 
 *  under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National 
 *  Laboratory (LANL), which is operated by Los Alamos National Security, LLC 
 *  for the U.S. Department of Energy. The U.S. Government has rights to use, 
 *  reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS 
 *  ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR 
 *  ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified
 *  to produce derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Los Alamos National Security, LLC, Los Alamos 
 *       National Laboratory, LANL, the U.S. Government, nor the names of its 
 *       contributors may be used to endorse or promote products derived from 
 *       this software without specific prior written permission.
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE LOS ALAMOS NATIONAL SECURITY, LLC AND 
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT 
 *  NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL
 *  SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *  
 *  CLAMR -- LA-CC-11-094
 *  
 *  Authors: Brian Atkinson          bwa@g.clemson.edu
             Bob Robey        XCP-2  brobey@lanl.gov
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <assert.h>
#include "PowerParser/PowerParser.hh"

#include "crux.h"
#include "timer/timer.h"
#include "fmemopen.h"

#ifdef HAVE_HDF5
#include "hdf5.h"
#endif
#ifdef HAVE_MPI
#include "mpi.h"
#endif

const bool CRUX_TIMING = true;
bool do_crux_timing = false;

#define RESTORE_NONE     0
#define RESTORE_RESTART  1
#define RESTORE_ROLLBACK 2

#ifndef DEBUG
#define DEBUG 0
#endif
#define DEBUG_RESTORE_VALS 1

using namespace std;
using PP::PowerParser;
// Pointers to the various objects.
PowerParser *parse;

char checkpoint_directory[] = "checkpoint_output";
int cp_num, rs_num;
int *backup;
void **crux_data;
size_t *crux_data_size;
#ifdef HAVE_HDF5
bool USE_HDF5 = true; //MSB
hid_t h5_fid;
herr_t h5err;

hid_t create_hdf5_parallel_file_plist();

void map_name_to_hdf5 (const char*, int, char*, int, char*, int);

void access_named_hdf5_values (const char *name, int name_size,
                              hsize_t rank, hsize_t *cur_size, 
                              void *values, hid_t datatype,
                              bool shared, bool store);
# ifdef HDF5_FF
hid_t e_stack;
hid_t tid1;
hid_t rid1, rid2;
hid_t rc_id;
hid_t h5_gid_b, h5_gid_d, h5_gid_m, h5_gid_s;
# endif

#endif


FILE *crux_time_fp;
struct timeval tcheckpoint_time;
struct timeval trestore_time;
int checkpoint_timing_count = 0;
float checkpoint_timing_sum = 0.0f;
float checkpoint_timing_size = 0.0f;
int rollback_attempt = 0;
FILE *store_fp, *restore_fp;
#ifdef HAVE_MPI
static MPI_File mpi_store_fp, mpi_restore_fp;
#endif
static int mype = 0, npes = 1;

Crux::Crux(int crux_type_in, int num_of_rollback_states_in, bool restart)
{
#ifdef HAVE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD,&mype);
   MPI_Comm_size(MPI_COMM_WORLD,&npes);
#endif

   num_of_rollback_states = num_of_rollback_states_in;
   crux_type = crux_type_in;
   checkpoint_counter = 0;

   if (crux_type != CRUX_NONE || restart){
      do_crux_timing = CRUX_TIMING;
      struct stat stat_descriptor;
      if (stat(checkpoint_directory,&stat_descriptor) == -1){
        mkdir(checkpoint_directory,0777);
      }
   }

   crux_data = (void **)malloc(num_of_rollback_states*sizeof(void *));
   for (int i = 0; i < num_of_rollback_states; i++){
      crux_data[i] = NULL;
   }
   crux_data_size = (size_t *)malloc(num_of_rollback_states*sizeof(size_t));


   if (do_crux_timing){
      char checkpointtimelog[60];
      sprintf(checkpointtimelog,"%s/crux_timing.log",checkpoint_directory);
      crux_time_fp = fopen(checkpointtimelog,"w");
   }
}

Crux::~Crux()
{
   for (int i = 0; i < num_of_rollback_states; i++){
      free(crux_data[i]);
   }
   free(crux_data);
   free(crux_data_size);

   if (do_crux_timing){
      if (checkpoint_timing_count > 0) {
         printf("CRUX checkpointing time averaged %f msec, bandwidth %f Mbytes/sec\n",
                checkpoint_timing_sum/(float)checkpoint_timing_count*1.0e3,
                checkpoint_timing_size/checkpoint_timing_sum*1.0e-6);

         fprintf(crux_time_fp,"CRUX checkpointing time averaged %f msec, bandwidth %f Mbytes/sec\n",
                checkpoint_timing_sum/(float)checkpoint_timing_count*1.0e3,
                checkpoint_timing_size/checkpoint_timing_sum*1.0e-6);

      fclose(crux_time_fp);
      }
   }
}

void Crux::store_MallocPlus(MallocPlus memory){
    malloc_plus_memory_entry *memory_item;

    int mype = 0;
    int npes = 1;

#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&mype);
    MPI_Comm_size(MPI_COMM_WORLD,&npes);
#endif

    for (memory_item = memory.memory_entry_by_name_begin(); 
            memory_item != memory.memory_entry_by_name_end();
            memory_item = memory.memory_entry_by_name_next() ){

        void *mem_ptr = memory_item->mem_ptr;
        if ((memory_item->mem_flags & RESTART_DATA) == 0) continue;

        if (DEBUG) {
            printf("MallocPlus ptr  %p: name %10s ptr %p dims %lu nelem (",
                    mem_ptr,memory_item->mem_name,memory_item->mem_ptr,memory_item->mem_ndims);

            char nelemstring[80];
            char *str_ptr = nelemstring;
            str_ptr += sprintf(str_ptr,"%lu", memory_item->mem_nelem[0]);
            for (uint i = 1; i < memory_item->mem_ndims; i++){
                str_ptr += sprintf(str_ptr,", %lu", memory_item->mem_nelem[i]);
            }
            printf("%12s",nelemstring);

            printf(") elsize %lu flags %d capacity %lu\n",
                    memory_item->mem_elsize,memory_item->mem_flags,memory_item->mem_capacity);
        }

#ifdef HAVE_HDF5
        if(USE_HDF5) {
	  printf("access_named_hdf5_values= %s\n",memory_item->mem_name);
            access_named_hdf5_values (memory_item->mem_name, 
                              strlen (memory_item->mem_name),
                              (hsize_t) memory_item->mem_ndims, 
                              (hsize_t *) memory_item->mem_nelem, 
                              mem_ptr, 
                              memory_item->mem_elsize == 4 ? 
                              H5T_NATIVE_INT : H5T_NATIVE_DOUBLE,
                              memory_item->mem_flags & REPLICATED_DATA, true);
        } else {
#endif
            int num_elements = 1;
            for (uint i = 0; i < memory_item->mem_ndims; i++){
                num_elements *= memory_item->mem_nelem[i];
            }
            store_field_header(memory_item->mem_name,30);
            if (memory_item->mem_flags & REPLICATED_DATA) { 
                if (memory_item->mem_elsize == 4){
                    store_replicated_int_array((int *)mem_ptr, num_elements);
                } else {
                    store_replicated_double_array((double *)mem_ptr, num_elements);
                }
            } else {
                if (memory_item->mem_elsize == 4){
                    store_int_array((int *)mem_ptr, num_elements);
                } else {
                    store_double_array((double *)mem_ptr, num_elements);
                }
            }
        }
#ifdef HAVE_HDF5   
    }
#endif   
}

void Crux::store_begin(size_t nsize, int ncycle)
{
#ifdef HDF5_FF
  uint64_t version;
  uint64_t trans_num;
  herr_t ret;
  size_t token_size[3];
  void *gset_token[3];
#endif

  int mype = 0;
  int npes = 1;

#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&mype);
    MPI_Comm_size(MPI_COMM_WORLD,&npes);
#endif

   cp_num = checkpoint_counter % num_of_rollback_states;
   cpu_timer_start(&tcheckpoint_time);

   if(crux_type == CRUX_IN_MEMORY) {
      if (crux_data[cp_num] != NULL) free(crux_data[cp_num]);
      crux_data[cp_num] = (int *)malloc(nsize);
      crux_data_size[cp_num] = nsize;
      store_fp = fmemopen(crux_data[cp_num], nsize, "w");
   } else if(crux_type == CRUX_DISK) {
      char backup_file[60];
#ifdef HAVE_HDF5
      if(USE_HDF5) {
          hid_t plist_id = create_hdf5_parallel_file_plist();
# ifdef HDF5_FF
	  sprintf(backup_file,"backup%05d.h5",ncycle);
# else
          sprintf(backup_file,"%s/backup%05d.h5",checkpoint_directory,ncycle);
# endif
#ifdef HDF5_FF
	  e_stack = H5EScreate();
	  assert(e_stack);

          if(!(h5_fid = H5Fcreate_ff(backup_file, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id, H5_EVENT_STACK_NULL)))
              printf("HDF5: Could not write HDF5 %s at iteration %d\n",backup_file,ncycle);

	  uint64_t version;
	  /* Acquire a read handle for container version 1 and create a read context. */
	  version = 1;
	  if ( 0 == mype ) {
	    rid1 = H5RCacquire( h5_fid, &version, H5P_DEFAULT, H5_EVENT_STACK_NULL ); 
	  } else {
	    rid1 = H5RCcreate( h5_fid, version); 
	  }
	  assert( rid1 >= 0 ); assert( version == 1 );
	  MPI_Barrier( MPI_COMM_WORLD );

	  /* create transaction object */
	  tid1 = H5TRcreate(h5_fid, rid1, (uint64_t)2);
	  assert(tid1); 

	  if(0 == mype) {

	    trans_num = 2;
	    ret = H5TRstart(tid1, H5P_DEFAULT, H5_EVENT_STACK_NULL);
	    assert(0 == ret);

	    // Leader also create some objects in transaction 1

	    // create group hierarchy 
	    if( (h5_gid_b = H5Gcreate_ff(h5_fid, "bootstrap", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT,tid1, e_stack) ) < 0) 
	      printf("HDF5: Could not create \"bootstrap\" group \n");
	    if( (h5_gid_d = H5Gcreate_ff(h5_fid, "default", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT,tid1, e_stack) ) < 0) 
	      printf("HDF5: Could not create \"default\" group \n");
	    if( (h5_gid_m = H5Gcreate_ff(h5_fid, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT,tid1, e_stack) ) < 0)
	      printf("HDF5: Could not create \"mesh\" group \n");
	    if( (h5_gid_s = H5Gcreate_ff(h5_fid, "state", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT,tid1, e_stack) ) < 0)
	      printf("HDF5: Could not create \"state\" group \n");

	    // Get token for group
	    ret = H5Oget_token(h5_gid_b, NULL, &token_size[0]);
	    assert(0 == ret);
	    ret = H5Oget_token(h5_gid_d, NULL, &token_size[1]);
	    assert(0 == ret);
	    ret = H5Oget_token(h5_gid_m, NULL, &token_size[2]);
	    assert(0 == ret);
	    ret = H5Oget_token(h5_gid_s, NULL, &token_size[3]);
	    assert(0 == ret);
	  
	    // allocate buffers for each token
	    gset_token[0] = malloc(token_size[0]);
	    gset_token[1] = malloc(token_size[1]);
	    gset_token[2] = malloc(token_size[2]);
	    gset_token[3] = malloc(token_size[3]);

	    // get the token buffer
	    ret = H5Oget_token(h5_gid_b, gset_token[0], &token_size[0]);
	    assert(0 == ret);
	    ret = H5Oget_token(h5_gid_d, gset_token[1], &token_size[1]);
	    assert(0 == ret);
	    ret = H5Oget_token(h5_gid_m, gset_token[2], &token_size[2]);
	    assert(0 == ret);
	    ret = H5Oget_token(h5_gid_s, gset_token[3], &token_size[3]);
	    assert(0 == ret);

	    // Broadcast token size
	    MPI_Bcast(&token_size, sizeof(size_t)*4, MPI_BYTE, 0, MPI_COMM_WORLD);

	    // Broadcast token
	    MPI_Bcast(gset_token[0], token_size[0], MPI_BYTE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(gset_token[1], token_size[1], MPI_BYTE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(gset_token[2], token_size[2], MPI_BYTE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(gset_token[3], token_size[3], MPI_BYTE, 0, MPI_COMM_WORLD);

	  } else {
	    // Receive token size
	    MPI_Bcast(&token_size, sizeof(size_t)*4, MPI_BYTE, 0, MPI_COMM_WORLD);

	    // Allocate token
	    gset_token[0] = malloc(token_size[0]);
	    gset_token[1] = malloc(token_size[1]);
	    gset_token[2] = malloc(token_size[2]);
	    gset_token[3] = malloc(token_size[3]);

	    // Receive token
	    MPI_Bcast(gset_token[0], token_size[0], MPI_BYTE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(gset_token[1], token_size[1], MPI_BYTE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(gset_token[2], token_size[2], MPI_BYTE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(gset_token[3], token_size[3], MPI_BYTE, 0, MPI_COMM_WORLD);

	    // Open group by token
	    h5_gid_b = H5Oopen_by_token(gset_token[0], tid1, e_stack);
	    h5_gid_d = H5Oopen_by_token(gset_token[1], tid1, e_stack);
	    h5_gid_m = H5Oopen_by_token(gset_token[2], tid1, e_stack);
	    h5_gid_s = H5Oopen_by_token(gset_token[3], tid1, e_stack);
	}
 	free(gset_token[0]);
 	free(gset_token[1]);
 	free(gset_token[2]);
 	free(gset_token[3]);
#else
	if(!(h5_fid = H5Fcreate(backup_file, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)))
	  printf("HDF5: Could not write HDF5 %s at iteration %d\n",backup_file,ncycle);
#endif
	H5Pclose(plist_id);
      } else {
#endif
          sprintf(backup_file,"%s/backup%05d.crx",checkpoint_directory,ncycle);
#ifdef HAVE_MPI
          int iret = MPI_File_open(MPI_COMM_WORLD, backup_file, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_store_fp);
          if(iret != MPI_SUCCESS) {
              printf("Could not write %s at iteration %d\n",backup_file,ncycle);
          }
#else
          store_fp = fopen(backup_file,"w");
          if(!store_fp){
              printf("Could not write %s at iteration %d\n",backup_file,ncycle);
          }
#endif
          if (mype == 0) {
              char symlink_file[60];
              sprintf(symlink_file,"%s/backup%1d.crx",checkpoint_directory,cp_num);
              symlink(backup_file, symlink_file);
              //      int ireturn = symlink(backup_file, symlink_file);
              //      if (ireturn == -1) {
              //         printf("Warning: error returned with symlink call for file %s and symlink %s\n",
              //                backup_file,symlink_file);
              //      }
          }
      }
#ifdef HAVE_HDF5
    }
#endif    
   if (do_crux_timing) {
      checkpoint_timing_size += nsize;
   }
   printf("finished store_begin \n"); // MSB
}

void Crux::store_field_header(const char *name, int name_size){
#ifdef HAVE_MPI
   assert(name != NULL);
   MPI_Status status;
   MPI_File_write_shared(mpi_store_fp, (void *)name, name_size, MPI_CHAR, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_CHAR, &count);
   printf("%d:Wrote %d characters at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif

#else
   assert(name != NULL && store_fp != NULL);
   fwrite(name,sizeof(char),name_size,store_fp);
#endif
}

#ifdef HAVE_HDF5
hid_t create_hdf5_parallel_file_plist()
{
    hid_t plist_id = H5P_DEFAULT;
#ifdef HAVE_MPI
    if( (plist_id = H5Pcreate(H5P_FILE_ACCESS)) < 0)
        printf("HDF5: Could not create property list \n");
    if( H5Pset_libver_bounds(plist_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST) < 0)
        printf("HDF5: Could set libver bounds \n");
# ifdef HDF5_FF
    H5Pset_fapl_iod(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
# else
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
# endif
#endif
    return plist_id;
}

void map_name_to_hdf5 (const char *name, int name_size,
                        char *group, int group_size,
                        char *label, int label_size)
{
    static const char * default_group = "default";
    int i, j;
    group[0] = '/';
    for (i=0; i<name_size; i++)
        if (name[i] == '_') break;
    if (i < name_size) {
        for (j=0; j<i; j++)
            group[1+j] = name[j];
        ++i;
    } else {
        for (j=0; default_group[j]; j++)
            group[1+j] = default_group[j];
        i=0;
    }    
    group[1+j] = '\0';
    for (j=i; name[j]; j++)
        label[j-i] = name[j];
    label[j-i] = '\0';    
}

void access_named_hdf5_values (const char *name, int name_size,
                              hsize_t rank, hsize_t *sizes, 
                              void *values, hid_t datatype,
                              bool shared, bool store)
{
    size_t length = 0, count = 1, offset = 0;
    char groupname[512], fieldname[512];
    hid_t hid_group, hid_space, hid_mem, hid_dataset, hid_plist = H5P_DEFAULT;
    int ret;
    map_name_to_hdf5(name, name_size, groupname, 512, fieldname, 512);
    for (int i=0; i<rank; i++)
        count *= sizes[i];
#ifdef HAVE_MPI
    hid_plist = H5Pcreate(H5P_DATASET_XFER);
#   ifdef HDF5_FF
    //    uint64_t trans_num;
    //printf("rid2 %ld \n", rid2);
    //tid1 = H5TRcreate(h5_fid, rid2, (uint64_t)2);
    //getchar();
    //trans_num = 3;
    //ret = H5TRstart(tid1, H5P_DEFAULT, H5_EVENT_STACK_NULL);
    //getchar();
#   else
    H5Pset_dxpl_mpio(hid_plist, H5FD_MPIO_COLLECTIVE);
#   endif
    if (npes > 1) {
        size_t *counts = new size_t[npes];
        MPI_Allgather (&count, sizeof(count), MPI_BYTE,
                       counts, sizeof *counts, MPI_BYTE,
                       MPI_COMM_WORLD);
        for (int i=0; i<npes; i++) {
            if (i == mype)
                offset = length;
            length += counts[i];
        }
        delete[] counts;
    } else {
#endif
        length = count;
#ifdef HAVE_MPI    
    }
#endif
#ifdef HDF5_FF
    htri_t exists;
    hid_t rc_id;
    size_t num_events = 0;

    if( (strstr(groupname,"bootstrap") !=NULL) ) {
      hid_group = h5_gid_b;
    } else if ( (strstr(groupname,"default") !=NULL) ) {
      hid_group = h5_gid_d;
    } else if ( (strstr(groupname,"mesh") !=NULL) ) {
	hid_group = h5_gid_m;
    } else if ( (strstr(groupname,"state") !=NULL) ) {
      hid_group = h5_gid_s;
    }

    if (!store)
    ;
    //  hid_group = H5Gopen_ff (h5_fid, groupname, H5P_DEFAULT, tid1, e_stack); // fix this rid2
    //}
#else    
    if (!store || H5Lexists(h5_fid, groupname, H5P_DEFAULT))
      hid_group = H5Gopen (h5_fid, groupname, H5P_DEFAULT);
#endif
    else
#ifdef HDF5_FF
    ;
#else
    hid_group = H5Gcreate (h5_fid, groupname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
    if (!hid_group) {
        fprintf(stderr, "Unable to create group: %30s\n", groupname);
        exit(1);
    }
    hid_mem = H5Screate_simple (1, (hsize_t *) &count, NULL);
    hid_space = H5Screate_simple (1, (hsize_t *) &length, NULL);
    if(!hid_space) {
        fprintf(stderr, "Unable to create space\n");
        exit(1);
    }
    if (store) {
#ifdef HDF5_FF

        hid_dataset = H5Dcreate_ff (hid_group, fieldname, datatype, hid_space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, tid1, e_stack);
#else
        hid_dataset = H5Dcreate (hid_group, fieldname, datatype, hid_space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif
    } else
#ifdef HDF5_FF
      hid_dataset = H5Dopen_ff (hid_group, fieldname, H5P_DEFAULT, tid1, e_stack);
#else
      hid_dataset = H5Dopen (hid_group, fieldname, H5P_DEFAULT);
#endif

    if (!hid_dataset) {
        fprintf(stderr, "Unable to create dataset: %30s\n", fieldname);
        exit(1);
    }
    herr_t status;
    status = H5Sselect_hyperslab (hid_space, H5S_SELECT_SET,
                                 (hsize_t *) &offset, NULL,
                                 (hsize_t *) &count, NULL);
    if(status < 0) {
        fprintf(stderr, "Unable to select correct hyperslab\n");
        exit(1);
    }
    if (store) {
#if HDF5_FF
      status = H5Dwrite_ff (hid_dataset, datatype, hid_mem, hid_space, hid_plist, values, tid1, e_stack);
      printf("finished write %d\n",status); }
#else
      status = H5Dwrite (hid_dataset, datatype, hid_mem, hid_space, hid_plist, values);
#endif
    else 
#if HDF5_FF
      status = H5Dread_ff (hid_dataset, datatype, hid_mem, hid_space, hid_plist, values, tid1, e_stack);
#else
      status = H5Dread (hid_dataset, datatype, hid_mem, hid_space, hid_plist, values);
#endif
#ifdef HDF5_FF
    H5Dclose_ff (hid_dataset, H5_EVENT_STACK_NULL);
    //    H5Gclose_ff (hid_group, e_stack);

//     H5ES_status_t status_es;
//     if(0 != mype) {
//       H5ESget_count(e_stack, &num_events);
//       H5ESwait_all(e_stack, &status_es);
//       H5ESclear(e_stack);
//       printf("%d events in event stack. Completion status = %d\n", num_events, status_es);
//       assert(status_es == H5ES_STATUS_SUCCEED);
//     }
//     // Leader process finished the transaction after all clients
//     // finish their updates. Leader also asks the library to acquire
//     // the committed transaction, that becomes a readable version
//     // after the commit completes.
//     MPI_Request mpi_req;
//     if(0 == mype) {
//       MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);

//       // make this synchronous so we know the container version has been acquired
//       ret = H5TRfinish(tid1, H5P_DEFAULT, &rid2, H5_EVENT_STACK_NULL);
//       assert(0 == ret);
//       ret = H5RCrelease(rid2, e_stack);
//     }
//     ret = H5TRclose(tid1);
//     assert(0 == ret);

//     // release container version 1. This is async.
//     if(0 == mype) {
//       ret = H5RCrelease(rid1, e_stack);
//       assert(0 == ret);
//     }

//     H5ESget_count(e_stack, &num_events);
//     H5ESwait_all(e_stack, &status_es);
//     printf("%d events in event stack. H5ESwait_all Completion status = %d\n", num_events, status_es);
//     H5ESclear(e_stack);
//     assert(status_es == H5ES_STATUS_SUCCEED);    
#else
    H5Dclose (hid_dataset);
    H5Gclose (hid_group);
#endif
    H5Sclose (hid_space);
    H5Sclose (hid_mem);
#ifdef HAVE_MPI
    H5Pclose (hid_plist);
#endif
}
#endif

void Crux::store_named_ints(const char *name, int name_size, int *int_vals, size_t nelem)
{
#ifdef HAVE_HDF5
    if (USE_HDF5) {
        access_named_hdf5_values (name, name_size, 1, (hsize_t *) &nelem, 
                                 int_vals, H5T_NATIVE_INT, false, true);

    } else {
#endif
        store_field_header (name, name_size);
        store_int_array (int_vals, nelem);
#ifdef HAVE_HDF5
    }
#endif    
}

void Crux::restore_named_ints(const char *name, int name_size, int *int_vals, size_t nelem)
{
#ifdef HAVE_HDF5
    if (USE_HDF5) {
        access_named_hdf5_values (name, name_size, 1, (hsize_t *) &nelem, 
                                 int_vals, H5T_NATIVE_INT, false, false);

    } else {
#endif
        char fname[512];
        restore_field_header (fname, name_size);
        restore_int_array (int_vals, nelem);
#ifdef HAVE_HDF5
    }
#endif    
}

void Crux::store_bools(bool *bool_vals, size_t nelem)
{
   assert(bool_vals != NULL && store_fp != NULL);
   fwrite(bool_vals,sizeof(bool),nelem,store_fp);
}

void Crux::store_ints(int *int_vals, size_t nelem)
{
   assert(int_vals != NULL && store_fp != NULL);
   fwrite(int_vals,sizeof(int),nelem,store_fp);
}

void Crux::store_longs(long long *long_vals, size_t nelem)
{
   assert(long_vals != NULL && store_fp != NULL);
   fwrite(long_vals,sizeof(long long),nelem,store_fp);
}

void Crux::store_sizets(size_t *size_t_vals, size_t nelem)
{
   assert(size_t_vals != NULL && store_fp != NULL);
   fwrite(size_t_vals,sizeof(size_t),nelem,store_fp);
}

void Crux::store_doubles(double *double_vals, size_t nelem)
{
   assert(double_vals != NULL && store_fp != NULL);
   fwrite(double_vals,sizeof(double),nelem,store_fp);
}

void Crux::store_int_array(int *int_array, size_t nelem)
{
#ifdef HAVE_MPI
   assert(int_array != NULL);
   MPI_Status status;
   MPI_File_write_shared(mpi_store_fp, int_array, (int)nelem, MPI_INT, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_INT, &count);
   printf("%d:Wrote %d integers at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif

#else
   assert(int_array != NULL && store_fp != NULL);
   fwrite(int_array,sizeof(int),nelem,store_fp);
#endif
}

void Crux::store_long_array(long long *long_array, size_t nelem)
{
   assert(long_array != NULL && store_fp != NULL);
   fwrite(long_array,sizeof(long long),nelem,store_fp);
}

void Crux::store_float_array(float *float_array, size_t nelem)
{
   assert(float_array != NULL && store_fp != NULL);
   fwrite(float_array,sizeof(float),nelem,store_fp);
}

void Crux::store_double_array(double *double_array, size_t nelem)
{
#ifdef HAVE_MPI
   assert(double_array != NULL);
   MPI_Status status;
   MPI_File_write_shared(mpi_store_fp, double_array, (int)nelem, MPI_DOUBLE, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_DOUBLE, &count);
   printf("%d:Wrote %d doubles at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif

#else
   assert(double_array != NULL && store_fp != NULL);
   fwrite(double_array,sizeof(double),nelem,store_fp);
#endif
}

void Crux::store_replicated_int_array(int *int_array, size_t nelem)
{
#ifdef HAVE_MPI
   assert(int_array != NULL);
   MPI_Status status;
   MPI_File_write_shared(mpi_store_fp, int_array, (int)nelem, MPI_INT, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_INT, &count);
   printf("%d:Wrote %d integers at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif

#else
   assert(int_array != NULL && store_fp != NULL);
   fwrite(int_array,sizeof(int),nelem,store_fp);
#endif
}

void Crux::store_replicated_double_array(double *double_array, size_t nelem)
{
#ifdef HAVE_MPI
   assert(double_array != NULL);
   MPI_Status status;
   MPI_File_write_shared(mpi_store_fp, double_array, (int)nelem, MPI_DOUBLE, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_DOUBLE, &count);
   printf("%d:Wrote %d doubles at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif

#else
   assert(double_array != NULL && store_fp != NULL);
   fwrite(double_array,sizeof(double),nelem,store_fp);
#endif
}

#ifdef HAVE_MPI
void Crux::store_distributed_int_array(int *int_array, size_t nelem, int flags)
{
   assert(int_array != NULL);
   //MPI_Datatype datatype = get_crux_datatype(DISTRIBUTED_INT_DATA);
   MPI_Status status;
   //MPI_File_write_shared(mpi_store_fp, int_array, nelem, MPI_INT, &status);
   printf("writing crux data type 8\n");
   //MPI_File_write_shared(mpi_store_fp, int_array, 1, crux_datatype[8], &status);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_INT, &count);
   printf("%d:Wrote %d integers at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif
}
void Crux::store_distributed_double_array(double *double_array, size_t nelem, int flags)
{
   assert(double_array != NULL);
   //MPI_Datatype datatype = get_crux_datatype(DISTRIBUTED_DOUBLE_DATA);
   MPI_Status status;
   //MPI_File_write_shared(mpi_store_fp, double_array, nelem, datatype, &status);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_DOUBLE_PRECISION, &count);
   printf("%d:Wrote %d doubles at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif
}
#endif

void Crux::store_end(void)
{
#ifdef HAVE_HDF5
   if(USE_HDF5) {
#if HDF5_FF
     H5ES_status_t status;
     size_t num_events = 0;
     herr_t ret;
     uint64_t version;

     /* none leader procs have to complete operations before notifying the leader */
    if(0 != mype) {
        H5ESget_count(e_stack, &num_events);
        H5ESwait_all(e_stack, &status);
        H5ESclear(e_stack);
        printf("%d events in event stack. Completion status = %d\n", num_events, status);
        assert(status == H5ES_STATUS_SUCCEED);
    }

    /* Barrier to make sure all processes are done writing so Process
       0 can finish transaction 1 and acquire a read context on it. */
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(0 == mype) {
      //  MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);

      // make this synchronous so we know the container version has been acquired
      ret = H5TRfinish(tid1, H5P_DEFAULT, &rid2, H5_EVENT_STACK_NULL);
      assert(0 == ret);
    }

    ret = H5TRclose(tid1);
    assert(0 == ret);
    /* release container version 1. This is async. */
    if(0 == mype) {
        ret = H5RCrelease(rid1, e_stack);
        assert(0 == ret);
    }
    H5ESget_count(e_stack, &num_events);
    H5ESwait_all(e_stack, &status);
    printf("%d events in event stack. H5ESwait_all Completion status = %d\n", num_events, status);
    H5ESclear(e_stack);
    assert(status == H5ES_STATUS_SUCCEED);

    if(H5Gclose_ff(h5_gid_b, e_stack) < 0)
      printf("HDF5: Could not close bootstrap group \n");
    if(H5Gclose_ff(h5_gid_d, e_stack) < 0)
      printf("HDF5: Could not close default group \n");
    if(H5Gclose_ff(h5_gid_m, e_stack) < 0)
      printf("HDF5: Could not close mesh group \n");
    if(H5Gclose_ff(h5_gid_s, e_stack) < 0)
      printf("HDF5: Could not close state group \n");

    /* Tell other procs that container version 2 is acquired */
    version = 2;
    MPI_Bcast(&version, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    /* other processes just create a read context object; no need to
       acquire it */
    if(0 != mype) {
      rid2 = H5RCcreate(h5_fid, version);
      assert(rid2 > 0);
    }
    /* close some objects */

    H5ESwait(e_stack, 0, &status);

    H5ESget_count(e_stack, &num_events);
    H5ESwait_all(e_stack, &status);
    printf("%d events in event stack. H5ESwait_all Completion status = %d\n", 
           num_events, status);
    H5ESclear(e_stack);

    MPI_Barrier(MPI_COMM_WORLD);    
    if(mype == 0) {
        ret = H5RCrelease(rid2, e_stack);
        assert(0 == ret);
    }


     if(H5Fclose_ff(h5_fid, 1, H5_EVENT_STACK_NULL) != 0)
       printf("HDF5: Could not close HDF5 file \n");

     H5ESget_count(e_stack, &num_events);

     H5EStest_all(e_stack, &status);
     printf("%d events in event stack. H5EStest_all Completion status = %d\n", num_events, status);
     
     H5ESwait_all(e_stack, &status);
     printf("%d events in event stack. H5ESwait_all Completion status = %d\n", num_events, status);
     
     ret = H5RCclose(rid1);
     assert(0 == ret);
     ret = H5RCclose(rid2);
     assert(0 == ret);
     
     H5ESclear(e_stack);
     
     ret = H5ESclose(e_stack);
     assert(ret == 0);  

#else
     if(H5Fclose(h5_fid) != 0)
       printf("HDF5: Could not close HDF5 file \n");
#endif
   } else {
#endif
#ifdef HAVE_MPI
       MPI_File_close(&mpi_store_fp);
#else
       assert(store_fp != NULL);
       fclose(store_fp);
#endif
#ifdef HAVE_HDF5
    }
#endif    

   double checkpoint_total_time = cpu_timer_stop(tcheckpoint_time);

   if (do_crux_timing){
      fprintf(crux_time_fp, "Total time for checkpointing was %g seconds\n", checkpoint_total_time);
      checkpoint_timing_count++;
      checkpoint_timing_sum += checkpoint_total_time;
   }

   checkpoint_counter++;
}

int restore_type = RESTORE_NONE;

void Crux::restore_MallocPlus(MallocPlus memory){
    char test_name[34];
    malloc_plus_memory_entry *memory_item;
    for (memory_item = memory.memory_entry_by_name_begin(); 
            memory_item != memory.memory_entry_by_name_end();
            memory_item = memory.memory_entry_by_name_next() ){
        void *mem_ptr = memory_item->mem_ptr;
        if ((memory_item->mem_flags & RESTART_DATA) == 0) continue;

        if (DEBUG) {
            printf("MallocPlus ptr  %p: name %10s ptr %p dims %lu nelem (",
                    mem_ptr,memory_item->mem_name,memory_item->mem_ptr,memory_item->mem_ndims);

            char nelemstring[80];
            char *str_ptr = nelemstring;
            str_ptr += sprintf(str_ptr,"%lu", memory_item->mem_nelem[0]);
            for (uint i = 1; i < memory_item->mem_ndims; i++){
                str_ptr += sprintf(str_ptr,", %lu", memory_item->mem_nelem[i]);
            }
            printf("%12s",nelemstring);

            printf(") elsize %lu flags %d capacity %lu\n",
                    memory_item->mem_elsize,memory_item->mem_flags,memory_item->mem_capacity);
        }
#ifdef HAVE_HDF5
        if(USE_HDF5) {
            access_named_hdf5_values (memory_item->mem_name, 
                    strlen (memory_item->mem_name),
                    (hsize_t) memory_item->mem_ndims, 
                    (hsize_t *) memory_item->mem_nelem, 
                    mem_ptr, 
                    memory_item->mem_elsize == 4 ? 
                    H5T_NATIVE_INT : H5T_NATIVE_DOUBLE,
                    memory_item->mem_flags & REPLICATED_DATA, false);
        } else {
#endif
            int num_elements = 1;
            for (uint i = 0; i < memory_item->mem_ndims; i++){
                num_elements *= memory_item->mem_nelem[i];
            }
            restore_field_header(test_name,30);
            if (strcmp(test_name,memory_item->mem_name) != 0) {
                printf("ERROR in restore checkpoint for %s %s\n",test_name,memory_item->mem_name);
#ifdef HAVE_MPI
                MPI_Finalize();
#endif
                exit(-1);
            }
            if (memory_item->mem_flags & REPLICATED_DATA) { 
                if (memory_item->mem_elsize == 4){
                    restore_replicated_int_array((int *)mem_ptr, num_elements);
                } else {
                    restore_replicated_double_array((double *)mem_ptr, num_elements);
                }
            } else {
                if (memory_item->mem_elsize == 4){
                    restore_int_array((int *)mem_ptr, num_elements);
                } else {
                    restore_double_array((double *)mem_ptr, num_elements);
                }
            }
        }
#ifdef HAVE_HDF5
    }
#endif    
}

void Crux::restore_begin(char *restart_file, int rollback_counter)
{
    rs_num = rollback_counter % num_of_rollback_states;

    cpu_timer_start(&trestore_time);

    if (restart_file != NULL){
        if (mype == 0) {
            printf("\n  ================================================================\n");
            printf(  "  Restoring state from disk file %s\n",restart_file);
            printf(  "  ================================================================\n\n");
        }
#ifdef HAVE_HDF5
# ifdef HDF5_FF
	hid_t last_rc_id;
	uint64_t last_version, v, version;
	hid_t file_id, fapl_id, rc_id;
# endif
        if (USE_HDF5) {
            hid_t plist_id = create_hdf5_parallel_file_plist();
#ifdef HDF5_FF
            if(!(h5_fid = H5Fopen_ff(restart_file, H5F_ACC_RDONLY, plist_id, &last_rc_id, H5_EVENT_STACK_NULL)))
                printf("HDF5: Could not restart from HDF5 file: %s\n", restart_file);

	    H5RCget_version( last_rc_id, &last_version );

	    v = last_version;
	    for ( v; v <= last_version; v++ ) {
	      version = v;
	      if ( 0 == mype ) {
		if(v < last_version) {
		  fprintf( stderr, "r%d: Try to acquire read context for cv %d\n", mype, (int)version );
		  rc_id = H5RCacquire( file_id, &version, H5P_DEFAULT, H5_EVENT_STACK_NULL );
		}
		else
		  rc_id = last_rc_id;
	      }
	      
	      MPI_Bcast( &rc_id, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
	      
	      if ( rc_id < 0 ) {
		if ( 0 == mype ) {
		  fprintf( stderr, "r%d: Failed to acquire read context for cv %d\n", mype, (int)v );
		}
	      } else {
		if ( 0 != mype ) {
		  rc_id = H5RCcreate( file_id, version ); 
		}
		assert ( version == v );
		//  print_container_contents_ff(file_id, rc_id, mype );
		
		MPI_Barrier( MPI_COMM_WORLD );
		if(v < last_version) {
		  if ( 0 == mype ) {
		    H5RCrelease( rc_id, H5_EVENT_STACK_NULL ); 
		  }
		  H5RCclose( rc_id ); 
		}
	      }
	    }

	    /* Release the read handle and close read context on cv obtained from H5Fopen_ff (by all ranks) */
	    H5RCrelease( last_rc_id, H5_EVENT_STACK_NULL );
	    H5RCclose( last_rc_id );
#else
            if(!(h5_fid = H5Fopen(restart_file, H5F_ACC_RDONLY, plist_id)))
                printf("HDF5: Could not restart from HDF5 file: %s\n", restart_file);
#endif
            H5Pclose(plist_id);
        } else {
#endif
#ifdef HAVE_MPI
            int iret = MPI_File_open(MPI_COMM_WORLD, restart_file, MPI_MODE_RDONLY | MPI_MODE_UNIQUE_OPEN, MPI_INFO_NULL, &mpi_restore_fp);
            if(iret != MPI_SUCCESS){
                //printf("Could not write %s at iteration %d\n",restart_file,crux_int_vals[8]);
                printf("Could not open restart file %s\n",restart_file);
            }
#else
            restore_fp = fopen(restart_file,"r");
            if(!restore_fp){
                //printf("Could not write %s at iteration %d\n",restart_file,crux_int_vals[8]);
                printf("Could not open restart file %s\n",restart_file);
            }
#endif
#ifdef HAVE_HDF5
        }
#endif    
        restore_type = RESTORE_RESTART;
    } else if(crux_type == CRUX_IN_MEMORY){
        printf("Restoring state from memory rollback number %d rollback_counter %d\n",rs_num,rollback_counter);
        restore_fp = fmemopen(crux_data[rs_num], crux_data_size[rs_num], "r");
        restore_type = RESTORE_ROLLBACK;
    } else if(crux_type == CRUX_DISK){
        char backup_file[60];

        sprintf(backup_file,"%s/backup%d.crx",checkpoint_directory,rs_num);
        printf("Restoring state from disk file %s rollback_counter %d\n",backup_file,rollback_counter);
        restore_fp = fopen(backup_file,"r");
        if(!restore_fp){
            //printf("Could not write %s at iteration %d\n",backup_file,crux_int_vals[8]);
            printf("Could not open restore file %s\n",backup_file);
        }
        restore_type = RESTORE_ROLLBACK;
    }
}

void Crux::restore_field_header(char *name, int name_size)
{
#ifdef HAVE_MPI
   assert(name != NULL);
   MPI_Status status;
   MPI_File_read_shared(mpi_restore_fp, name, name_size, MPI_CHAR, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_CHAR, &count);
   printf("%d:Read %d characters at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif

#else
   int name_read = fread(name,sizeof(char),name_size,restore_fp);
   if (name_read != name_size){
      printf("Warning: number of elements read %d is not equal to request %d\n",name_read,name_size);
   }
#endif
}

void Crux::restore_bools(bool *bool_vals, size_t nelem)
{
   size_t nelem_read = fread(bool_vals,sizeof(bool),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
}

void Crux::restore_ints(int *int_vals, size_t nelem)
{
   size_t nelem_read = fread(int_vals,sizeof(int),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
}

void Crux::restore_longs(long long *long_vals, size_t nelem)
{
   size_t nelem_read = fread(long_vals,sizeof(long),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
}

void Crux::restore_sizets(size_t *size_t_vals, size_t nelem)
{
   size_t nelem_read = fread(size_t_vals,sizeof(size_t),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
}

void Crux::restore_doubles(double *double_vals, size_t nelem)
{
   size_t nelem_read = fread(double_vals,sizeof(double),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
}

int *Crux::restore_int_array(int *int_array, size_t nelem)
{
#ifdef HAVE_MPI
   assert(int_array != NULL);
   MPI_Status status;
   MPI_File_read_shared(mpi_restore_fp, int_array, (int)nelem, MPI_INT, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_INT, &count);
   printf("%d:Read %d integers at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif

#else
   size_t nelem_read = fread(int_array,sizeof(int),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
#endif
   return(int_array);
}

long long *Crux::restore_long_array(long long *long_array, size_t nelem)
{
   size_t nelem_read = fread(long_array,sizeof(long long),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
   return(long_array);
}

float *Crux::restore_float_array(float *float_array, size_t nelem)
{
   size_t nelem_read = fread(float_array,sizeof(float),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
   return(float_array);
}

double *Crux::restore_double_array(double *double_array, size_t nelem)
{
#ifdef HAVE_MPI
   MPI_Status status;
   MPI_File_read_shared(mpi_restore_fp, double_array, (int)nelem, MPI_DOUBLE, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_DOUBLE, &count);
   printf("%d:Read %d doubles at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif
  
#else
   size_t nelem_read = fread(double_array,sizeof(double),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
#endif
   return(double_array);
}

int *Crux::restore_replicated_int_array(int *int_array, size_t nelem)
{
#ifdef HAVE_MPI
   assert(int_array != NULL);
   MPI_Status status;
   MPI_File_read_shared(mpi_restore_fp, int_array, (int)nelem, MPI_INT, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_INT, &count);
   printf("%d:Read %d integers at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif

#else
   size_t nelem_read = fread(int_array,sizeof(int),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
#endif
   return(int_array);
}

double *Crux::restore_replicated_double_array(double *double_array, size_t nelem)
{
#ifdef HAVE_MPI
   MPI_Status status;
   MPI_File_read_shared(mpi_restore_fp, double_array, (int)nelem, MPI_DOUBLE, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_DOUBLE, &count);
   printf("%d:Read %d doubles at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif
  
#else
   size_t nelem_read = fread(double_array,sizeof(double),nelem,restore_fp);
   if (nelem_read != nelem){
      printf("Warning: number of elements read %lu is not equal to request %lu\n",nelem_read,nelem);
   }
#endif
   return(double_array);
}

#ifdef HAVE_MPI
int *Crux::restore_distributed_int_array(int *int_array, size_t nelem, int flags)
{
   assert(int_array != NULL);
   //MPI_Datatype datatype = get_crux_datatype(DISTRIBUTED_INT_DATA);
   MPI_Status status;
   //MPI_File_read_shared(mpi_restore_fp, int_array, (int)nelem, datatype, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_INT, &count);
   printf("%d:Read %d integers at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif

   return(int_array);
}

double *Crux::restore_distributed_double_array(double *double_array, size_t nelem, int flags)
{
   //MPI_Datatype datatype = get_crux_datatype(DISTRIBUTED_DOUBLE_DATA);
   MPI_Status status;
   //MPI_File_read_shared(mpi_restore_fp, double_array, (int)nelem, datatype, &status);
   MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_RESTORE_VALS
   int count;
   MPI_Get_count(&status, MPI_DOUBLE, &count);
   printf("%d:Read %d doubles at line %d in file %s\n",mype,count,__LINE__,__FILE__);
#endif
  
   return(double_array);
}
#endif

void Crux::restore_end(void)
{
   double restore_total_time = cpu_timer_stop(trestore_time);

   if (do_crux_timing){
      if (restore_type == RESTORE_RESTART) {
         fprintf(crux_time_fp, "Total time for restore was %g seconds\n", restore_total_time);
      } else if (restore_type == RESTORE_ROLLBACK){
         fprintf(crux_time_fp, "Total time for rollback %d was %g seconds\n", rollback_attempt, restore_total_time);
      }
   }
#ifdef HAVE_HDF5
   if(USE_HDF5) {
     if(H5Fclose(h5_fid) != 0) {
       printf("HDF5: Could not close HDF5 file!!\n");
     }
   } else {
#endif
#ifdef HAVE_MPI
       MPI_File_close(&mpi_store_fp);
#else
       assert(restore_fp != NULL);
       fclose(restore_fp);
#endif
#ifdef HAVE_HDF5
    }
#endif
}

int Crux::get_rollback_number()
{
  rollback_attempt++;
  return(checkpoint_counter % num_of_rollback_states);
}

void Crux::set_crux_type(int crux_type_in)
{
  crux_type = crux_type_in;
}

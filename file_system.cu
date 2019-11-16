#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
// newly added:
#include <string.h>

__device__ __managed__ u32 gtime = 0;

__device__ __managed__ u32* Sort_indexer;
__device__ __managed__ u32* Sort_timer;
__device__ __managed__ u32* Sort_size;
__device__ __managed__ u32* Sort_creation;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
	// init variables
	fs->volume = volume;

	// init constants
	fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
	fs->FCB_SIZE = FCB_SIZE;
	fs->FCB_ENTRIES = FCB_ENTRIES;
	fs->STORAGE_SIZE = VOLUME_SIZE;
	fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
	fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
	fs->MAX_FILE_NUM = MAX_FILE_NUM;
	fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
	fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
	// initial set all uchar to be 0xFF.
	for (int i = 0; i < VOLUME_SIZE; i++) {
		fs->volume[i] = 255;   // change all bit to invalid
	}
	for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
		fs->volume[i] = 0;  // change the bitmap to be free initially
	}
}

__device__ int byteToInt(uchar a, uchar b) {
	int result = int((a << 8) + b);
	return result;
}

__device__ int Sort_cmp_S(const void* a_, const void* b_) {														// the function prepared for qsort.
	const int* a = (int*)a_;
	const int* b = (int*)b_;
	if (Sort_size[*a]==Sort_size[*b])
	{
		if (Sort_creation[*a]> Sort_creation[*b])
		{
			return 1;
		}
		else {
			return -1;
		}
	}
	else if(Sort_size[*a]>Sort_size[*b]){
		return -1;
	}
	else {
		return 1;
	}
}

__device__ int Sort_cmp_D(const void* a_, const void* b_) {														// the function prepared for qsort.
	const int* a = (int*)a_;
	const int* b = (int*)b_;
	if (Sort_timer[*a] == Sort_timer[*b])
	{
		return 0;
	}
	else if (Sort_timer[*a] > Sort_timer[*b]) {
		return -1;
	}
	else {
		return 1;
	}
}

// implementation of qsort in cuda:

__device__ void cuda_swap(void* p1, void* p2, int size) {
	int i = 0;
	for (i = 0; i < size; i++)
	{
		char tmp = *((char*)p1 + i);
		*((char*)p1 + i) = *((char*)p2 + i);
		*((char*)p2 + i) = tmp;
	}
}

__device__ void cuda_qsort(void* base, int count, int size, int(*cmp)(const void*, const void*)) {
	int i = 0;
	int j = 0;
	for (i = 0; i <count-1; i++)
	{
		for (j=0; j < count-i-1; j++)
		{
			if (cmp((char*)base+j*size,(char*)base+(j+1)*size)>0)
			{
				cuda_swap((char*)base + j * size, (char*)base + (j + 1)*size, size);
			}
		}
	}
}

__device__ void Sorter(u32 op, u32 existing_file_number) {
	//printf("		[Sorter accessed]\n");
	if (op == LS_D)
	{
		//printf("			[Signal:op == LS_D]\n");
		cuda_qsort(Sort_indexer, existing_file_number, sizeof(int), Sort_cmp_D);
	}
	else if (op == LS_S) {
		//printf("			[Signal:op == LS_S]\n");
		cuda_qsort(Sort_indexer, existing_file_number, sizeof(int), Sort_cmp_S);
	}
	else {
		//printf("[Operation Error]: Wrong usage!\n");
		// need to modify: to raise error.
		return;
	}
}

// smaller helper functions:
__device__ u32 cuda_strlen(char* a) {
	u32 result = 0;
	for (u32 i = 0; i < 50; i++)
	{
		if (a[i] == '\0')
		{
			break;
		}
		result++;
	}
	return result;
}

__device__ u32 cuda_strcmp(char* a, char* b) {
	int len_a = cuda_strlen(a);
	int len_b = cuda_strlen(b);
	if (len_a != len_b)
	{
		return 1;
	}
	for (int i = 0; i < len_a; i++)
	{
		if (a[i]!=b[i])
		{
			return 1;
		}
	}
	return 0;
}

__device__ void cuda_strncpy(char* dest, char* src, u32 n) {
	for (u32 i = 0; i < n; i++)
	{
		dest[i] = src[i];
	}
}

__device__ void cuda_strcpy(char* dest, char* src) {
	u32 len = cuda_strlen(src);
	for (u32 i = 0; i < len; i++)
	{
		dest[i] = src[i];
	}
	dest[len] = '\0';
	//printf("			[strcpy]:%s\n",dest);
}

// functions for garbage collection
// firstly, three helper functions:
__device__ u32 blockindex2FCBindex(FileSystem *fs, u32 blockindex) {
	/*u32 block_address = fs->FILE_BASE_ADDRESS + blockindex * fs->STORAGE_BLOCK_SIZE;*/
	u32 FCB_index = 0xFFFFFFFF;
	for (u32 i = 0; i < fs->FCB_ENTRIES; i++) {
		u32 iteration_FCB_address = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*i;										// access the 1st byte in the ith entry in 1024 FCB entries.
		uchar iteration_file_block_pointer_first_half = fs->volume[iteration_FCB_address + 28];
		uchar iteration_file_block_pointer_second_half = fs->volume[iteration_FCB_address + 29];
		u32 iteration_file_block_index = byteToInt(iteration_file_block_pointer_first_half, iteration_file_block_pointer_second_half);
		if (blockindex == iteration_FCB_address)
		{
			FCB_index = i;
			break;
		}
	}
	return FCB_index;
}

__device__ int sizeInBlock(FileSystem* fs, int size) {
	int sizeInBlock = 0;
	if (size % fs->STORAGE_BLOCK_SIZE != 0) {
		sizeInBlock = size / fs->STORAGE_BLOCK_SIZE + 1;
	}
	else {
		sizeInBlock = size / fs->STORAGE_BLOCK_SIZE;
	}
	return sizeInBlock;
	// file size in storage blks
}

__device__ u32 FCBindex2FileSize(FileSystem *fs, u32 FCBindex) {
	u32 FCB_address = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*FCBindex;
	uchar file_size_first_half = fs->volume[FCB_address + 26];
	uchar  file_size_second_half = fs->volume[FCB_address + 27];
	u32 file_size = byteToInt(file_size_first_half, file_size_second_half);
	return file_size;
}


// main utility function:

__device__ u32 SearchSpaceRoutine(FileSystem *fs, u32 size) {
	// the 'size' here stands for the number of needed blocks!
	//printf("		[SearchSpaceRoutine accessed]\n");
	u32 pos = -1;
	// the result to return us actually the file block index.
	u32 count = 0;
	u32 undecided_flag = 1;
	u32 success_flag = 0;
	uchar* uchar_list = fs->volume;
	for (u32 index = 0; index < fs->SUPERBLOCK_SIZE; index++)
	{
		//printf("		Signalindex= %d\n",index);
		uchar tmpList[8];
		u32 tmp = uchar_list[index];
		//printf("		tmp= %d\n", tmp);
		for (int i = 8 - 1; i >= 0; i--)
		{
			u32 d = tmp % 2;
			if (d == 0)
			{
				tmpList[i] = '0';
				//printf("		tmp[%d]= %c\n", i, tmpList[i]);
			}
			else {
				tmpList[i] = '1';
				//printf("		tmp[%d]= %c\n", i, tmpList[i]);
			}
			tmp = tmp / 2;
		}
		//printf("done");
		//printf("		Signalindex= %d Signal1\n", index);
		for (u32 offset = 0; offset < 8; offset++)
		{
			if (undecided_flag == 1 && tmpList[offset] == '0')
			{
				pos = index * 8 + offset;
				undecided_flag = 0;
			}
			if (undecided_flag == 0 && tmpList[offset] == '0') {
				count++;
			}
			if (undecided_flag == 0 && tmpList[offset] == '1') {
				count = 0;
				undecided_flag = 1;
			}
			if (count == size)
			{
				success_flag = 1;
				break;
			}
		}
		if (success_flag == 1)
		{
			break;
		}
	}
	if (success_flag == 1)
	{
		//printf("		Signal found\n");
		return pos;
	}
	//printf("		Signal unfound\n");
	return -1;
}

// where we always set the char_list as fs->volume.
__device__ void SuperBlockModifier(FileSystem *fs, u32 first_file_block_index, u32 len, u32 op) {
	// the 'len' here stands for the number of needed blocks!
	//printf("		[SuperBlockModifier accessed]\n");
	uchar* uchar_list = fs->volume;
	if (first_file_block_index + len - 1 > 32767)																					// since the index of empty list is from 0 to 32k-1=32767.
	{
		printf("Out of bounds!\n");
		// need to modify: to raise error.
		return;
	}
	/*u32 first_byte_index = first_file_block_index / 8;
	u32 first_byte_offset = first_file_block_index % 8;
	u32 second_byte_index = (first_file_block_index+len) / 8;
	u32 second_byte_offset = (first_file_block_index+len) % 8;*/
	int first_byte_index = first_file_block_index / 8;
	int first_byte_offset = first_file_block_index % 8;
	int second_byte_index = (first_file_block_index + len) / 8;
	int second_byte_offset = (first_file_block_index + len) % 8;
	if (op == 0)
	{
		//printf("		Signal1\n");
		if (first_byte_index == second_byte_index)
		{
			uchar tool = 0xFF;
			for (u32 i = (7 - second_byte_offset + 1); i <= (7 - first_byte_offset); i++)
			{
				tool = tool - (1 << i);
			}
			uchar_list[first_byte_index] = (uchar_list[first_byte_index] & tool);
		}
		else {
			uchar tool1 = 0xFF;
			uchar tool2 = 0xFF;
			for (u32 i = 0; i <= (7 - first_byte_offset); i++)
			{
				tool1 = tool1 - (1 << i);
			}
			uchar_list[first_byte_index] = (uchar_list[first_byte_index] & tool1);

			for (u32 byte_index = first_byte_index + 1; byte_index < second_byte_index; byte_index++)
			{
				uchar_list[byte_index] = 0x00;
			}

			for (int i = 8 - 1; i > (7 - second_byte_offset); i--)
			{
				tool2 = tool2 - (1 << i);
			}
			uchar_list[second_byte_index] = (uchar_list[second_byte_index] & tool2);
		}
	}
	else {
		if (first_byte_index == second_byte_index)
		{
			uchar tool = 0x00;
			for (u32 i = (7 - second_byte_offset + 1); i <= (7 - first_byte_offset); i++)
			{
				tool = tool + (1 << i);
			}
			uchar_list[first_byte_index] = (uchar_list[first_byte_index] | tool);
		}
		else {
			uchar tool1 = 0x00;
			uchar tool2 = 0x00;
			for (u32 i = 0; i <= (7 - first_byte_offset); i++)
			{
				tool1 = tool1 + (1 << i);
			}
			uchar_list[first_byte_index] = (uchar_list[first_byte_index] | tool1);

			for (u32 byte_index = first_byte_index + 1; byte_index < second_byte_index; byte_index++)
			{
				uchar_list[byte_index] = 0xFF;
			}

			for (int i = 8 - 1; i > (7 - second_byte_offset); i--)
			{
				tool2 = tool2 + (1 << i);
			}
			uchar_list[second_byte_index] = (uchar_list[second_byte_index] | tool2);
		}
	}
}


__device__ void moveRoutine(FileSystem *fs, u32 last_0_index, u32 next_1_index, u32 size) {
	for (u32 index = 0; index < size; index++)
	{
		u32 iteration_last_0_address = fs->FILE_BASE_ADDRESS + (last_0_index + index) * fs->STORAGE_BLOCK_SIZE;
		u32 iteration_next_1_address = fs->FILE_BASE_ADDRESS + (next_1_index + index) * fs->STORAGE_BLOCK_SIZE;
		for (u32 offset = 0; offset < fs->STORAGE_BLOCK_SIZE; offset++)
		{
			fs->volume[iteration_last_0_address + offset] = fs->volume[iteration_next_1_address + offset];
		}
	}
	return;
}

__device__ void ResizeRoutine(FileSystem *fs) {
	u32 last_0_index = 0;
	u32 next_1_index = 0;
	u32 find_0_first_time_flag = 1;
	//
	uchar* uchar_list = fs->volume;
	//
	for (int index = 0; index < fs->SUPERBLOCK_SIZE; index++)
	{
		uchar tmpList[8];
		u32 tmp = uchar_list[index];
		for (int i = 8 - 1; i >= 0; i--)
		{
			u32 d = tmp % 2;
			if (d == 0)
			{
				tmpList[i] = '0';
				//printf("		tmp[%d]= %c\n", i, tmpList[i]);
			}
			else {
				tmpList[i] = '1';
				//printf("		tmp[%d]= %c\n", i, tmpList[i]);
			}
			tmp = tmp / 2;
		}
		for (u32 offset = 0; offset < 8; offset++)
		{
			if (find_0_first_time_flag == 1 && tmpList[offset] == '1')
			{
				last_0_index++;
			}
			if (find_0_first_time_flag == 1 && tmpList[offset] == '0')
			{
				find_0_first_time_flag = 0;
				next_1_index = last_0_index;
				next_1_index++;
			}
			if (find_0_first_time_flag == 0 && tmpList[offset] == '1')
			{
				u32 FCB_index_of_file_to_move = blockindex2FCBindex(fs, next_1_index);
				u32 FCB_address_of_file_to_move = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*FCB_index_of_file_to_move;
				u32 size_of_file_to_move = FCBindex2FileSize(fs, FCB_index_of_file_to_move);
				// refresh FCB pointer to last_0_index;
				fs->volume[FCB_address_of_file_to_move + 28] = uchar(last_0_index >> 8);
				fs->volume[FCB_address_of_file_to_move + 29] = uchar(last_0_index & 0x000000FF);
				//
				moveRoutine(fs, last_0_index, next_1_index, size_of_file_to_move);
				SuperBlockModifier(fs, last_0_index, size_of_file_to_move, 1);
				SuperBlockModifier(fs, next_1_index, size_of_file_to_move, 0);
				next_1_index++;
			}
			if (find_0_first_time_flag == 0 && tmpList[offset] == '0')
			{
				next_1_index++;
			}
		}
	}
}


__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	int fcbIdx = 0;
	if (cuda_strlen(s) > 20) {
		printf("Wrong file name inpiut >> File Name Size Exceed 20!\n");
		return -1;
	}
	int foundFlag = 0;
	/*
	iterating through all the FCB entries to find the corresponding blocks of the filename given
	*/
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		int filenameFlag = -1;
		int filenameCounter = 0;
		for (char *p = s; *p; ++p) {
			if (*p != fs->volume[i * 32 + fs->SUPERBLOCK_SIZE + filenameCounter]) {
				filenameFlag = 1;
			}
			else {
				filenameCounter++;
			}
		}
		if (filenameFlag == -1) {
			//printf("found file name\n");
			fcbIdx = i;
			foundFlag = 1;
			break;
		}

	}
	// if found in FCB entries:
	if (foundFlag == 1) {
		int fcbAddr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE* fcbIdx;
		// to modify the read/write mode as uchar(op).
		fs->volume[fcbAddr + 24] = uchar(op);
		return fcbIdx;		// we can only return the FCB entry index.
	}
	// if not found in FCB entries:
	// if op == READ, raise error:
	if (op == G_READ) {
		printf("Cannot find the file to read!\n");
		return 1;
	}
	if (op == G_WRITE) {
		// if op == WRITE:
		// 1. first allocate the first free block that is available
		int fcbIdx = -1; // performing fake allocation
		int hasEmptyFCB = 0;
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			uchar blkPtr_firstHalf = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*i + 28];
			uchar blkPtr_secondHalf = fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*i + 29];
			if (blkPtr_firstHalf == 0xFF && blkPtr_secondHalf == 0xFF)
			{
				fcbIdx = i;
				hasEmptyFCB = 1;
				break;
			}
		}
		// step2: return the empty FCB entry index.
		if (hasEmptyFCB == 1) {
			// step2.1: set the file name
			int address = fcbIdx * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
			for (int i = 0; i < 20; i++) {
				fs->volume[address + i] = s[i];
			}
			// to modify the read/write mode as uchar(op).
			fs->volume[address + 24] = uchar(op);
			return fcbIdx;
		}
		// step3: if not found, raise error, which means no enough space.
		printf("No More FCB Available!\n");
	}

	return -1;
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
	int address = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*fp;
	uchar blkPtr_firstHalf = fs->volume[address + 28];
	uchar blkPtr_secondHalf = fs->volume[address + 29];

	int blkIdx = byteToInt(blkPtr_firstHalf, blkPtr_secondHalf);
	int blkAddr = fs->FILE_BASE_ADDRESS + blkIdx * fs->STORAGE_BLOCK_SIZE;
	if (fs->volume[address + 24] != 0) {
		printf("Performing read in write mode!\n");
		return;
	}
	for (int i = 0; i < size; i++) {
		output[i] = fs->volume[blkAddr + i];
	}

}


__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
	//printf("	[fs_write accessed]\n");
	if (size > 1024) {
		printf("File Size Larger Than 1024 Bytes!\n");
		return -1;
	}
	int isNewAllocation = 0;
	int FCB_Addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*fp;

	uchar blkIdxFirst = fs->volume[FCB_Addr + 28];
	uchar blkIdxSecond = fs->volume[FCB_Addr + 29];

	int originalBlkIdx = byteToInt(blkIdxFirst, blkIdxSecond);
	// u32 originalBlkAddr = fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES*fs->FCB_SIZE + originalBlkIdx*fs->STORAGE_BLOCK_SIZE;
	int originalBlkAddr = fs->FILE_BASE_ADDRESS + originalBlkIdx * fs->STORAGE_BLOCK_SIZE;
	// check read/write mode:
	if (fs->volume[FCB_Addr+24] != 1)
	{
		printf("Write in Read Mode!\n");
		return -1;
	}
	// decide whether newly allocated.
	if (blkIdxFirst == 255 && blkIdxSecond == 255)
	{
		isNewAllocation = 1;
	}
	//printf("	[Signal2]\n");
	// if not newly allocated:
	if (!isNewAllocation)
	{
		//printf("	[Signal3]\n");
		// step1: check size:
		int blocksNeeded = sizeInBlock(fs,size);
		int oldSize = byteToInt(fs->volume[FCB_Addr + 26], fs->volume[FCB_Addr + 27]);											// 20+6;
		int oldBlkNumber = sizeInBlock(fs,oldSize);
		int newBlkIdx = -1;
		int newBlkAddr = -1;
		
		int blkChanged = 0;
		if (size > oldSize) {
			blkChanged = 1;
			newBlkIdx = SearchSpaceRoutine(fs, blocksNeeded);
		}
		
		// step2: if not enough, call file resizing routine:
		if (newBlkIdx == -1)
		{
			ResizeRoutine(fs);
			newBlkIdx = SearchSpaceRoutine(fs, blocksNeeded);
			if (newBlkIdx == -1)															// which means no possible total empty space.
			{
				printf("No More Block Available in the FileSystem \n");
				return -1;
			}
		}
		//printf("		[Signal3.2]\n");
		// step3:overwrite the file:
		newBlkAddr = fs->FILE_BASE_ADDRESS + newBlkIdx * fs->STORAGE_BLOCK_SIZE;
		for (int i = 0; i < size; i++)
		{
			// store the content of the file into volume
			fs->volume[newBlkAddr + i] = input[i];
		}
		//printf("		[Signal3.3]\n");
		// step4: if file block changed, refresh the super block info (the empty list):
		if (blkChanged)
		{
			SuperBlockModifier(fs, originalBlkIdx, oldBlkNumber, 0);
			SuperBlockModifier(fs, newBlkIdx, blocksNeeded, 1);
			// step5: refresh the FCB info2 ( pointer):
			fs->volume[FCB_Addr + 28] = uchar(newBlkIdx >> 8);
			fs->volume[FCB_Addr + 29] = uchar(newBlkIdx & 0x000000FF);
		}
		// step5: refresh the FCB info1 ( new size):
		fs->volume[FCB_Addr + 26] = uchar(size >> 8);
		fs->volume[FCB_Addr + 27] = uchar(size & 0x000000FF);
		//printf("		[Signal3.5]\n");
		// step5: refresh the FCB info (access time, write time):
		fs->volume[FCB_Addr + 20] = uchar(gtime>>8);														// then the access time must < 1<<17, need to modify;
		fs->volume[FCB_Addr + 21] = uchar(gtime & 0x000000FF);
		gtime++;
		//printf("		[Signal3.6]\n");
	}
	// if newly allocated:
	else
	{
		//printf("	[Signal4]\n");
		int new_size_needed = size;
		int blocksNeeded = sizeInBlock(fs,new_size_needed);
		int newBlkIdx = 0xFFFFFFFF;
		int newBlkAddr = 0xFFFFFFFF;
		// step1: call file resizing routine:
		newBlkIdx = SearchSpaceRoutine(fs, blocksNeeded);
		//printf("		[Signal4.0]\n");
		if (newBlkIdx == 0xFFFFFFFF)
		{
			//printf("		[Signal4.1]\n");
			ResizeRoutine(fs);
			newBlkIdx = SearchSpaceRoutine(fs, blocksNeeded);
			if (newBlkIdx == 0xFFFFFFFF)															// which means no possible total empty space.
			{
				printf("No More Block Available, FileSystem is full\n");
				return -1;
			}
		}
		//printf("		[Signal4.2]\n");
		// step2: overwrite the file:
		newBlkAddr = fs->FILE_BASE_ADDRESS + newBlkIdx * fs->STORAGE_BLOCK_SIZE;
		for (u32 i = 0; i < size; i++)
		{
			fs->volume[newBlkAddr + i] = input[i];
		}
		//printf("		[Signal4.3]\n");
		// step3: refresh the FCB info (creation time, access time, write time, new size):
		SuperBlockModifier(fs, newBlkIdx, blocksNeeded, 1);
		//printf("		[Signal4.4]\n");
		// step4: refresh the FCB info1 ( new size):
		fs->volume[FCB_Addr + 26] = uchar(new_size_needed >> 8);
		fs->volume[FCB_Addr + 27] = uchar(new_size_needed & 0x000000FF);
		// step5: refresh the FCB info2 ( pointer):
		fs->volume[FCB_Addr + 28] = uchar(newBlkIdx >> 8);
		fs->volume[FCB_Addr + 29] = uchar(newBlkIdx & 0x000000FF);
		// step5: refresh the FCB info (modified time, creation time):
		fs->volume[FCB_Addr + 20] = uchar(gtime >> 8);														// then the access time must < 1<<17, need to modify;
		fs->volume[FCB_Addr + 21] = uchar(gtime & 0x000000FF);
		fs->volume[FCB_Addr + 22] = uchar(gtime >> 8);														// then the access time must < 1<<17, need to modify;
		fs->volume[FCB_Addr + 23] = uchar(gtime & 0x000000FF);
		gtime++;
		//printf("		[Signal4.5]\n");
	}
	return 0;
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	//printf("	[fs_gsys accessed]\n");
	uchar all_File_Names[1024][20];
	u32 all_File_Modified_Time[1024];
	u32 all_File_Size[1024];
	u32 all_File_index[1024];
	u32 all_File_Creation[1024];
	u32 existing_file_number = 0;
	// step1: firstly find the all existing files with existing_file_number as number, and store their names, dates and sizes.
	// initialize the indexer first:
	//printf("		[Signal1]\n");
	for (u32  i = 0; i < 1024; i++)
	{
		all_File_index[i] = i;
	}
	//printf("		[Signal2]\n");
	for (u32 FCB_index = 0; FCB_index < fs->FCB_ENTRIES; FCB_index++)
	{
		u32 FCB_Addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*FCB_index;
		uchar blkIdxFirst = fs->volume[FCB_Addr + 28];
		uchar blkIdxSecond = fs->volume[FCB_Addr + 29];
		if (blkIdxFirst == 0xFF && blkIdxSecond == 0xFF) {
			continue;
		}
		//printf("		[Signal2]:%s\n", (char*)&(fs->volume[FCB_Addr]));
		cuda_strncpy((char*)&all_File_Names[existing_file_number], (char*)&(fs->volume[FCB_Addr]), 20);
		all_File_Modified_Time[existing_file_number] = byteToInt(fs->volume[FCB_Addr + 20], fs->volume[FCB_Addr + 21]);
		all_File_Size[existing_file_number] = byteToInt(fs->volume[FCB_Addr + 26], fs->volume[FCB_Addr + 27]);
		all_File_Creation[existing_file_number] = byteToInt(fs->volume[FCB_Addr + 22], fs->volume[FCB_Addr + 23]);
		//printf("	Signal2:name:%s,time:%d\n", (char*)&all_File_Names[existing_file_number],all_File_Modified_Time[existing_file_number]);
		existing_file_number = existing_file_number+1;
	}
	//printf("		[Signal3]:%d\n", existing_file_number);
	Sort_indexer = all_File_index;
	Sort_size = all_File_Size;
	Sort_timer = all_File_Modified_Time;
	Sort_creation = all_File_Creation;
	//printf("		[Signal4]\n");
	// step2: if sort and print by dates:
	if (op == LS_D)
	{
		//printf("		[Signal5:op == LS_D]\n");
		Sorter(LS_D, existing_file_number);
	}// step3: if sort and print by sizes:
	else if (op == LS_S)
	{
		//printf("		[Signal6:op == LS_S]\n");
		Sorter(LS_S, existing_file_number);
	}
	// neither: must be wrong.
	else {
		//printf("[Operation Error]: Wrong usage!\n");
		// need to modify: to raise error.
	}
	// step3: print out.
	//printf("		[Signal7]\n");
	if (op == LS_D)
	{
		printf("===sort by modified time===\n");
	}
	else
	{
		printf("===sort by file size===\n");
	}
	//printf("		[Signal8: existing_file_number=%d]\n", existing_file_number);
	for (u32 i = 0; i < existing_file_number; i++)
	{
		u32 index_in_char_list = all_File_index[i];
		// the order changement of the indexer will not change the timer and sizer, whic means only indexer changed.
		if (op == LS_D)
		{
			printf("%s\n",all_File_Names[index_in_char_list]);
		}
		else
		{
			printf("%s\t%d\n", all_File_Names[index_in_char_list],all_File_Size[index_in_char_list]);
		}
	}
	//printf("		[Signal9]\n");
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	if (cuda_strlen(s) > 20)
	{
		printf("[Input Size Error]: File Name Size Exceed 20!\n");
		return;
	}
	u32 find_name_flag = 0;
	u32 found_FCB_index = 0xFFFFFFFF;
	for (u32 i = 0; i < fs->FCB_ENTRIES; i++) {
		u32 iteration_FCB_address = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*i;										// access the 1st byte in the ith entry in 1024 FCB entries.
		// uchar* iteration_file_name = &(fs->volume[iteration_FCB_address]);										// since the file name is the first info, offset here is 0.
		char iteration_file_name[21];
		cuda_strncpy(iteration_file_name, (char*)&(fs->volume[iteration_FCB_address]), 20);							// after test, use (char*) to explictly trans uchar* to char*.
		iteration_file_name[20] = '\0';
		if (cuda_strcmp(s, (char*)iteration_file_name) == 0) {
			found_FCB_index = i;
			find_name_flag = 1;
			break;
		}
	}
	if (find_name_flag != 1)
	{
		printf("[User Operation Error]: Removing Unexisting File!\n");
		return;
		// need to modify: to raise error.
	}
	u32 remove_FCB_address = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*found_FCB_index;
	u32 remove_FCB_size = byteToInt(fs->volume[remove_FCB_address + 26], fs->volume[remove_FCB_address + 27]);
	u32 remove_FCB_file_block_number = sizeInBlock(fs,remove_FCB_size);
	// remove-step1: remove FCB.
	for (u32 i = 0; i < 32; i++)
	{
		fs->volume[remove_FCB_address + i] = 0xFF;
	}
	// remove-step2: remove super-block.
	SuperBlockModifier(fs, found_FCB_index, remove_FCB_file_block_number, 0);

}

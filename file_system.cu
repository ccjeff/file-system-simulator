#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ __managed__ u32 gtime = 0;

__device__ __managed__ int* global_time;
__device__ __managed__ int* global_size;
__device__ __managed__ int* global_created;

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

// acting as a comparator

__device__ int cmpSize(const void* a_, const void* b_) {		// the function prepared for qsort.
	const int* a = (int*)a_;
	const int* b = (int*)b_;
	if (global_size[*a] == global_size[*b])
	{
		if (global_created[*a] > global_created[*b])
		{
			return 1;
		}
		else {
			return -1;
		}
	}
	else if (global_size[*a] > global_size[*b]) {
		return -1;
	}
	else {
		return 1;
	}
}

__device__ int cmpDate(const void* a_, const void* b_) {
	const int* a = (int*)a_;
	const int* b = (int*)b_;
	if (global_time[*a] == global_time[*b])
	{
		return 0;
	}
	else if (global_time[*a] > global_time[*b]) {
		return -1;
	}
	else {
		return 1;
	}
}

__device__ void swap(void* p1, void* p2, int size) {
	int i = 0;
	for (i = 0; i < size; i++)
	{
		char tmp = *((char*)p1 + i);
		*((char*)p1 + i) = *((char*)p2 + i);
		*((char*)p2 + i) = tmp;
	}
}


// REFERENCE:: C source code of Qsort
__device__ void sorting(void* base, int count, int size, int(*cmp)(const void*, const void*)) {
	int i = 0;
	int j = 0;
	for (i = 0; i < count - 1; i++)
	{
		for (j = 0; j < count - i - 1; j++)
		{
			if (cmp((char*)base + j * size, (char*)base + (j + 1)*size) > 0)
			{
				swap((char*)base + j * size, (char*)base + (j + 1)*size, size);
			}
		}
	}
}



__device__ void getSortedArr(int op, int fileCounter, int* creationTime, 
	int* size, int* time, uchar filenames[1024][20], int* allIdx) {
	
	if (op == LS_D)
	{
		//printf("			[Signal:op == LS_D]\n");
		sorting(allIdx, fileCounter, sizeof(int), cmpDate);
	}
	else if (op == LS_S) {
		//printf("			[Signal:op == LS_S]\n");
		sorting(allIdx, fileCounter, sizeof(int), cmpSize);
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


// functions for garbage collection
// firstly, three helper functions:
__device__ int blkIdxToFCBIdx(FileSystem *fs, u32 blockindex) {
	/*u32 block_address = fs->FILE_BASE_ADDRESS + blockindex * fs->STORAGE_BLOCK_SIZE;*/
	int FCB_index = -1;
	for (int i = 0; i < fs->FCB_ENTRIES; i++) {
		int fcbAddr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*i;			// access the 1st byte in the ith entry in 1024 FCB entries.
		uchar iteration_file_block_pointer_first_half = fs->volume[fcbAddr + 28];
		uchar iteration_file_block_pointer_second_half = fs->volume[fcbAddr + 29];
		u32 iteration_file_block_index = byteToInt(iteration_file_block_pointer_first_half, iteration_file_block_pointer_second_half);
		if (blockindex == fcbAddr) {
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

__device__ u32 FCBIdxToFileSize(FileSystem *fs, u32 FCBindex) {
	u32 FCB_address = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*FCBindex;
	uchar file_size_first_half = fs->volume[FCB_address + 26];
	uchar  file_size_second_half = fs->volume[FCB_address + 27];
	u32 file_size = byteToInt(file_size_first_half, file_size_second_half);
	return file_size;
}


__device__ u32 findNextFree(FileSystem *fs, u32 size) {
	// the 'size' here stands for the number of needed blocks!
	//printf("		[findNextFree accessed]\n");
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


__device__ void changeSuperBlock(FileSystem *fs, u32 blkIdx, u32 len, u32 op) {
	
	/*
	if op == 1, change the superblock to be 'in use' and vice versa
	the 'len' here stands for the number of needed blocks
	*/
	
	if (blkIdx + len - 1 > 32767)																					// since the index of empty list is from 0 to 32k-1=32767.
	{
		printf("The block index is out of bound\n");
		return;
	}
	int first_byte_index = blkIdx / 8;
	int first_byte_offset = blkIdx % 8;
	int second_byte_index = (blkIdx + len) / 8;
	int second_byte_offset = (blkIdx + len) % 8;

	if (op == 0) {
		// free the blocks in use
		if (first_byte_index == second_byte_index)
		{
			uchar tool = 255;
			for (int i = (7 - second_byte_offset + 1); i <= (7 - first_byte_offset); i++)
			{
				tool = tool - (1 << i);
			}
			fs->volume[first_byte_index] = (fs->volume[first_byte_index] & tool);
		}
		else {
			uchar tool1 = 255;
			uchar tool2 = 255;
			for (int i = 0; i <= (7 - first_byte_offset); i++)
			{
				tool1 = tool1 - (1 << i);
			}
			fs->volume[first_byte_index] = (fs->volume[first_byte_index] & tool1);

			for (int byte_index = first_byte_index + 1; byte_index < second_byte_index; byte_index++)
			{
				fs->volume[byte_index] = 0;
			}

			for (int i = 7; i > (7 - second_byte_offset); i--)
			{
				tool2 = tool2 - (1 << i);
			}
			fs->volume[second_byte_index] = (fs->volume[second_byte_index] & tool2);
		}
	}
	else {
		if (first_byte_index == second_byte_index) {
			uchar tool = 0;
			for (int i = (7 - second_byte_offset + 1); i <= (7 - first_byte_offset); i++)
			{
				tool = tool + (1 << i);
			}
			fs->volume[first_byte_index] = (fs->volume[first_byte_index] | tool);
		}
		else {
			uchar tool1 = 0;
			uchar tool2 = 0;
			for (int i = 0; i <= (7 - first_byte_offset); i++)
			{
				tool1 = tool1 + (1 << i);
			}
			fs->volume[first_byte_index] = (fs->volume[first_byte_index] | tool1);

			for (int byte_index = first_byte_index + 1; byte_index < second_byte_index; byte_index++)
			{
				fs->volume[byte_index] = 255;
			}

			for (int i = 7; i > (7 - second_byte_offset); i--)
			{
				tool2 = tool2 + (1 << i);
			}
			fs->volume[second_byte_index] = (fs->volume[second_byte_index] | tool2);
		}
	}
}


__device__ void moveBlocks(FileSystem *fs, u32 last_0, u32 next_1, u32 size) {
	for (u32 index = 0; index < size; index++)
	{
		int last_0_address = fs->FILE_BASE_ADDRESS + (last_0 + index) * fs->STORAGE_BLOCK_SIZE;
		int next_1_address = fs->FILE_BASE_ADDRESS + (next_1 + index) * fs->STORAGE_BLOCK_SIZE;
		for (int offset = 0; offset < fs->STORAGE_BLOCK_SIZE; offset++)
		{
			fs->volume[last_0_address + offset] = fs->volume[next_1_address + offset];
		}
	}
	return;
}

__device__ void ResizeRoutine(FileSystem *fs) {
	int last_0_index = 0;
	int next_1_index = 0;
	int find_0_first_time_flag = 1;
	
	//
	for (int index = 0; index < fs->SUPERBLOCK_SIZE; index++)
	{
		uchar tmpList[8];
		int tmp = fs->volume[index];
		for (int i = 7; i >= 0; i--)
		{
			int d = tmp % 2;
			if (d == 0)
			{
				tmpList[i] = '0';
			}
			else {
				tmpList[i] = '1';
			}
			tmp = tmp / 2;
		}
		for (int offset = 0; offset < 8; offset++)
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
				int FCB_index_of_file_to_move = blkIdxToFCBIdx(fs, next_1_index);
				int FCB_address_of_file_to_move = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*FCB_index_of_file_to_move;
				int size_of_file_to_move = FCBIdxToFileSize(fs, FCB_index_of_file_to_move);
				// refresh FCB pointer to last_0_index;
				fs->volume[FCB_address_of_file_to_move + 28] = uchar(last_0_index >> 8);
				fs->volume[FCB_address_of_file_to_move + 29] = uchar(last_0_index & 0x000000FF);
				//
				moveBlocks(fs, last_0_index, next_1_index, size_of_file_to_move);
				changeSuperBlock(fs, last_0_index, size_of_file_to_move, 1);
				changeSuperBlock(fs, next_1_index, size_of_file_to_move, 0);
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
	if (size > 1024) {
		printf("File Size Larger Than 1024 Bytes!\n");
		return -1;
	}
	int isNewAllocation = 0;
	int FCB_Addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*fp;

	uchar blkIdxFirst = fs->volume[FCB_Addr + 28];
	uchar blkIdxSecond = fs->volume[FCB_Addr + 29];

	int originalBlkIdx = byteToInt(blkIdxFirst, blkIdxSecond);
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
	// if not newly allocated:
	if (!isNewAllocation)
	{
		// step1: check size:
		int blocksNeeded = sizeInBlock(fs,size);
		int oldSize = byteToInt(fs->volume[FCB_Addr + 26], fs->volume[FCB_Addr + 27]);		// 20+6;
		int oldBlkNumber = sizeInBlock(fs,oldSize);
		int newBlkIdx = -1;
		int newBlkAddr = -1;
		
		int blkChanged = 0;
		if (size > oldSize) {
			blkChanged = 1;
			newBlkIdx = findNextFree(fs, blocksNeeded);
		}
		
		// step2: if not enough, call file resizing routine:
		if (newBlkIdx == -1)
		{
			// do collection of the fragments
			ResizeRoutine(fs);
			newBlkIdx = findNextFree(fs, blocksNeeded);
			if (newBlkIdx == -1)						// which means no possible total empty space.
			{
				printf("No More Block Available in the FileSystem \n");
				return -1;
			}
		}
		// step3:overwrite the file:
		newBlkAddr = fs->FILE_BASE_ADDRESS + newBlkIdx * fs->STORAGE_BLOCK_SIZE;
		for (int i = 0; i < size; i++)
		{
			// store the content of the file into volume
			fs->volume[newBlkAddr + i] = input[i];
		}
		// step4: if file block changed, refresh the super block info (the empty list):
		if (blkChanged) {
			changeSuperBlock(fs, originalBlkIdx, oldBlkNumber, 0);
			changeSuperBlock(fs, newBlkIdx, blocksNeeded, 1);
			// step5: refresh the FCB info2 ( pointer):
			fs->volume[FCB_Addr + 28] = uchar(newBlkIdx >> 8);
			fs->volume[FCB_Addr + 29] = uchar(newBlkIdx & 0x000000FF);
		}
		// step5: refresh the FCB info1 ( new size):
		fs->volume[FCB_Addr + 26] = uchar(size >> 8);
		fs->volume[FCB_Addr + 27] = uchar(size & 0x000000FF);

		// step5: refresh the FCB info (access time, write time):
		fs->volume[FCB_Addr + 20] = uchar(gtime>>8);	// then the access time must < 1<<17, need to modify;
		fs->volume[FCB_Addr + 21] = uchar(gtime & 0x000000FF);
		gtime++;
	}
	else
	{
		// if newly allocated:
		int new_size_needed = size;
		int blocksNeeded = sizeInBlock(fs,new_size_needed);
		int newBlkIdx = -1;
		int newBlkAddr = -1;
		// step1: call file resizing routine:
		newBlkIdx = findNextFree(fs, blocksNeeded);
		if (newBlkIdx == -1)
		{
			//printf("		[Signal4.1]\n");
			ResizeRoutine(fs);
			newBlkIdx = findNextFree(fs, blocksNeeded);
			if (newBlkIdx == -1)	// which means no possible total empty space.
			{
				printf("No More Block Available, FileSystem is full\n");
				return -1;
			}
		}
		// step2: overwrite the file:
		newBlkAddr = fs->FILE_BASE_ADDRESS + newBlkIdx * fs->STORAGE_BLOCK_SIZE;
		for (int i = 0; i < size; i++)
		{
			fs->volume[newBlkAddr + i] = input[i];
		}
		// step3: refresh the FCB info (creation time, access time, write time, new size):
		changeSuperBlock(fs, newBlkIdx, blocksNeeded, 1);
		// step4: refresh the FCB info1 ( new size):
		fs->volume[FCB_Addr + 26] = uchar(new_size_needed >> 8);
		fs->volume[FCB_Addr + 27] = uchar(new_size_needed & 0x000000FF);
		// step5: refresh the FCB info2 ( pointer):
		fs->volume[FCB_Addr + 28] = uchar(newBlkIdx >> 8);
		fs->volume[FCB_Addr + 29] = uchar(newBlkIdx & 0x000000FF);
		// step5: refresh the FCB info (modified time, creation time):
		fs->volume[FCB_Addr + 20] = uchar(gtime >> 8);
		fs->volume[FCB_Addr + 21] = uchar(gtime & 0x000000FF);
		fs->volume[FCB_Addr + 22] = uchar(gtime >> 8);
		fs->volume[FCB_Addr + 23] = uchar(gtime & 0x000000FF);
		gtime++;
	}
	return 0;
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	uchar allFilenames[1024][20];
	int modifiedTimes[1024];
	int allSizes[1024];
	int allIndex[1024];
	int createdTime[1024];
	int fileCounter = 0;
	// step1: firstly find the all existing files with fileCounter as number, and store their names, dates and sizes.
	// initialize the index first:
	for (int i = 0; i < 1024; i++) {
		allIndex[i] = i;
	}
	for (int FCB_index = 0; FCB_index < fs->FCB_ENTRIES; FCB_index++) {
		// iterating through all FCBs to find the used blocks
		int FCB_Addr = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*FCB_index;
		uchar blkIdxFirst = fs->volume[FCB_Addr + 28];
		uchar blkIdxSecond = fs->volume[FCB_Addr + 29];
		if (blkIdxFirst == 255 && blkIdxSecond == 255) {
			// blk unused
			continue;
		}
		for (int i = 0; i < 20; i++) {
			allFilenames[fileCounter][i] = fs->volume[FCB_Addr + i];
		}

		modifiedTimes[fileCounter] = byteToInt(fs->volume[FCB_Addr + 20], fs->volume[FCB_Addr + 21]);
		allSizes[fileCounter] = byteToInt(fs->volume[FCB_Addr + 26], fs->volume[FCB_Addr + 27]);
		createdTime[fileCounter] = byteToInt(fs->volume[FCB_Addr + 22], fs->volume[FCB_Addr + 23]);
		
		fileCounter++;
	}
	
	global_size = allSizes;
	global_time = modifiedTimes;
	global_created = createdTime;

	// step2: if sort and print by dates:
	if (op == LS_D)
	{
		getSortedArr(LS_D, fileCounter,createdTime,allSizes,modifiedTimes,allFilenames,allIndex);
	}// step3: if sort and print by sizes:
	else if (op == LS_S)
	{
		getSortedArr(LS_S, fileCounter, createdTime, allSizes, modifiedTimes, allFilenames, allIndex);
	}
	// neither: must be wrong.
	else {
		// need to modify: to raise error.
	}
	// step3: print out.
	if (op == LS_D)
	{
		printf("===sort by modified time===\n");
	}
	else
	{
		printf("===sort by file size===\n");
	}
	
	for (int i = 0; i < fileCounter; i++)
	{
		int index_in_char_list = allIndex[i];
		// the order changement of the indexer will not change the timer and sizer, whic means only indexer changed.
		if (op == LS_D)
		{
			printf("%s\n",allFilenames[index_in_char_list]);
		}
		else
		{
			printf("%s\t%d\n", allFilenames[index_in_char_list],allSizes[index_in_char_list]);
		}
	}
	
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	if (op == RM) {
		if (cuda_strlen(s) > 20)
		{
			printf("File Name Size Exceed The Largest Filename Limit \n");
			return;
		}
		int foundFile = 0;
		int found_FCB_index = -1;
		for (int i = 0; i < fs->FCB_ENTRIES; i++) {
			int iteration_FCB_address = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*i;
			// access the 1st byte in the ith entry in 1024 FCB entries.

			char filename[21];
			for (int i = 0; i < 20; i++) {
				// setup filename
				filename[i] = fs->volume[iteration_FCB_address + i];
			}
			filename[20] = '\0';

			if (cuda_strcmp(s, (char*)filename) == 0) {
				found_FCB_index = i;
				foundFile = 1;
				break;
			}
		}
		if (foundFile != 1)
		{
			printf("Removing Unexisting File \n");
			return;
		}
		int remove_FCB_address = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*found_FCB_index;
		int remove_FCB_size = byteToInt(fs->volume[remove_FCB_address + 26], fs->volume[remove_FCB_address + 27]);
		int remove_FCB_file_block_number = sizeInBlock(fs, remove_FCB_size);
		// remove-step1: remove FCB.
		for (int i = 0; i < fs->FCB_SIZE; i++)
		{
			fs->volume[remove_FCB_address + i] = 255;
		}
		// remove-step2: remove super-block.
		changeSuperBlock(fs, found_FCB_index, remove_FCB_file_block_number, 0);
	}

}

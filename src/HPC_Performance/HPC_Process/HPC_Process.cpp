#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#include "sys/times.h"

//#include "sys/vtimes.h"

static clock_t lastCPU, lastSysCPU, lastUserCPU;
static unsigned long long lastTotalUser, lastTotalUserLow, lastTotalSys, lastTotalIdle;
static int numProcessors;

typedef void *HANDLE;

void InitCurrentProcess(){
    FILE* file;
    struct tms timeSample;
    char line[128];

    lastCPU = times(&timeSample);
    lastSysCPU = timeSample.tms_stime;
    lastUserCPU = timeSample.tms_utime;

    file = fopen("/proc/cpuinfo", "r");
    numProcessors = 0;
    while(fgets(line, 128, file) != NULL){
        if (strncmp(line, "processor", 9) == 0) numProcessors++;
    }
    fclose(file);
}


double GetCurrentValueFromCurrentProcess(){
    struct tms timeSample;
    clock_t now;
    double percent;

    now = times(&timeSample);
    if (now <= lastCPU || timeSample.tms_stime < lastSysCPU ||
        timeSample.tms_utime < lastUserCPU){
        //Overflow detection. Just skip this value.
        percent = -1.0;
    }
    else{
        percent = (timeSample.tms_stime - lastSysCPU) +(timeSample.tms_utime - lastUserCPU);
        percent /= (now - lastCPU);
        percent /= numProcessors;
        percent *= 100;
    }
    lastCPU = now;
    lastSysCPU = timeSample.tms_stime;
    lastUserCPU = timeSample.tms_utime;

    return percent;
}



void InitCurrentlyCPU(){
    FILE* file = fopen("/proc/stat", "r");
    fscanf(file, "cpu %llu %llu %llu %llu", &lastTotalUser, &lastTotalUserLow,
        &lastTotalSys, &lastTotalIdle);
    fclose(file);
}

double GetCurrentValueFromCPU(){
    double percent;
    FILE* file;
    unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;

    file = fopen("/proc/stat", "r");
    fscanf(file, "cpu %llu %llu %llu %llu", &totalUser, &totalUserLow,
        &totalSys, &totalIdle);
    fclose(file);

    if (totalUser < lastTotalUser || totalUserLow < lastTotalUserLow ||
        totalSys < lastTotalSys || totalIdle < lastTotalIdle){
        //Overflow detection. Just skip this value.
        percent = -1.0;
    }
    else{
        total = (totalUser - lastTotalUser) + (totalUserLow - lastTotalUserLow) +
            (totalSys - lastTotalSys);
        percent = total;
        total += (totalIdle - lastTotalIdle);
        percent /= total;
        percent *= 100;
    }

    lastTotalUser = totalUser;
    lastTotalUserLow = totalUserLow;
    lastTotalSys = totalSys;
    lastTotalIdle = totalIdle;

    return percent;
}

/*
double GetPercentUsed()
{

    static HANDLE self;
    static int numProcessors;
    SYSTEM_INFO sysInfo;

    double percent;

    numProcessors = sysInfo.dwNumberOfProcessors;

    //Getting system times information
    FILETIME SysidleTime;
    FILETIME SyskernelTime; 
    FILETIME SysuserTime; 
    ULARGE_INTEGER SyskernelTimeInt, SysuserTimeInt;
    GetSystemTimes(&SysidleTime, &SyskernelTime, &SysuserTime);
    memcpy(&SyskernelTimeInt, &SyskernelTime, sizeof(FILETIME));
    memcpy(&SysuserTimeInt, &SysuserTime, sizeof(FILETIME));
    __int64 denomenator = SysuserTimeInt.QuadPart + SyskernelTimeInt.QuadPart;  

    //Getting process times information
    FILETIME ProccreationTime, ProcexitTime, ProcKernelTime, ProcUserTime;
    ULARGE_INTEGER ProccreationTimeInt, ProcexitTimeInt, ProcKernelTimeInt, ProcUserTimeInt;
    GetProcessTimes(self, &ProccreationTime, &ProcexitTime, &ProcKernelTime, &ProcUserTime);
    memcpy(&ProcKernelTimeInt, &ProcKernelTime, sizeof(FILETIME));
    memcpy(&ProcUserTimeInt, &ProcUserTime, sizeof(FILETIME));
    __int64 numerator = ProcUserTimeInt.QuadPart + ProcKernelTimeInt.QuadPart;
    //QuadPart represents a 64-bit signed integer (ULARGE_INTEGER)

    percent = 100*(numerator/denomenator);
    return(percent);
}

*/


/*
void Info()
{
//Total Virtual Memory:
    struct sysinfo memInfo;

    sysinfo (&memInfo);
    long long totalVirtualMem = memInfo.totalram;
    //Add other values in next statement to avoid int overflow on right hand side...
    totalVirtualMem += memInfo.totalswap;
    totalVirtualMem *= memInfo.mem_unit;

    //Virtual Memory currently used:
    long long virtualMemUsed = memInfo.totalram - memInfo.freeram;
    //Add other values in next statement to avoid int overflow on right hand side...
    virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
    virtualMemUsed *= memInfo.mem_unit;

    //Total Physical Memory (RAM):
    long long totalPhysMem = memInfo.totalram;
    //Multiply in next statement to avoid int overflow on right hand side...
    totalPhysMem *= memInfo.mem_unit;

    //Physical Memory currently used:
    long long physMemUsed = memInfo.totalram - memInfo.freeram;
    //Multiply in next statement to avoid int overflow on right hand side...
    physMemUsed *= memInfo.mem_unit;
}
*/


/*
int GetValueVirtualMemoryCurrentlyUsedByCurrentProcess()
{ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}
*/


//Virtual Memory currently used by current process:

int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

int getValue(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}


int main(int argc, char** argv) {
    double Value;
    int v;

    InitCurrentProcess();
    Value=GetCurrentValueFromCurrentProcess();
    printf("GetCurrentValueFromCurrentProcess=%d\n",Value);
    InitCurrentlyCPU();
    Value=GetCurrentValueFromCPU();
    printf("GetCurrentValueFromCPU=%d\n",Value);
    //Value=GetPercentUsed();
    //printf("%d\n",Value);
    //void Info();
    //v=GetValueVirtualMemoryCurrentlyUsedByCurrentProcess();
    //printf("%i\n",Value);
    v=getValue();
    printf("Proc Status=%i\n",Value);

    return 0;
}
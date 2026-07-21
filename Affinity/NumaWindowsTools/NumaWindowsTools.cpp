/*
MIT License
Copyright (c) 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


// this cpp needs the Windows lib: SetupAPI.lib

#include "NumaWindowsTools.h"

#include <initguid.h>
#include <windows.h>
#include <setupapi.h>
#include <cfgmgr32.h>
#include <devpkey.h>
#include <vector>
#include <string>
#include <devpropdef.h>
#include <devguid.h>
#include <stdio.h>
#include <algorithm>


namespace NumaWindowsTool
{
	
	// desc in header
	unsigned int CountSystemNUMAnodes()
	{
		// Ask for classic NUMA nodes
		DWORD bytes = 0;
		if (GetLogicalProcessorInformationEx(RelationNumaNode, nullptr, &bytes) ||
			GetLastError() != ERROR_INSUFFICIENT_BUFFER)
			return false;

		std::vector<BYTE> buf(bytes);
		if (!GetLogicalProcessorInformationEx(
			RelationNumaNode,
			reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
			&bytes))
			return false;

		// Count RelationNumaNode
		size_t count = 0;
		for (BYTE *p = buf.data(), *e = buf.data() + bytes; p < e; ) {
			auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
			if (info->Relationship == RelationNumaNode || info->Relationship == RelationNumaNodeEx)
				++count;
			p += info->Size;
		}

		return (unsigned int)count;
	}

	// example:  convert the string "PCI segment 1 bus 2, device 3, function 4"  into the vector: [1,2,3,4]
	std::vector<uint32_t> extract_uints(const std::wstring& s)
	{
		std::vector<uint32_t> out;

		uint32_t val = 0;
		bool in = false;

		size_t i = 0;
		while (i < s.size()) 
		{
			wchar_t ch = s[i++];
			if (ch >= L'0' && ch <= L'9') {
				uint32_t d = (uint32_t)(ch - L'0');
				val = in ? (val * 10u + d) : d;
				in = true;
			} else {
				if (in) { 
					out.push_back(val); 
					in = false; 
				}
			}
		}
		if (in) 
			out.push_back(val);
		return out;
	}


	// get a 'DEVPROP_TYPE_STRING' key of a device
	bool get_str_prop(HDEVINFO set, SP_DEVINFO_DATA &dev, const DEVPROPKEY &key, std::wstring &out)
	{
		DEVPROPTYPE type = DEVPROP_TYPE_EMPTY;
		DWORD sz = 0;

		// first, query required size ( in bytes )
		if (!SetupDiGetDevicePropertyW(set, &dev, &key, &type, nullptr, 0, &sz, 0) &&
			GetLastError() != ERROR_INSUFFICIENT_BUFFER)
			return false;

		if (type != DEVPROP_TYPE_STRING)
			return false;

		// sz includes the trailing L'\0'. Convert bytes to wchar count.
		size_t wc_count = sz / sizeof(wchar_t);
		if (wc_count == 0) { 
			out.clear(); return true; 
		}

		out.resize(wc_count);

		// Fill buffer directly into the wstring storage.
		if (!SetupDiGetDevicePropertyW(set, &dev, &key, &type, reinterpret_cast<PBYTE>(&out[0]), sz, &sz, 0))
			return false;

		return true;
	}

	// get DEVPROP_TYPE_UINT32 key of a device
	bool get_u32_prop(HDEVINFO set, SP_DEVINFO_DATA &dev, const DEVPROPKEY &key, uint32_t *out)
	{
		DEVPROPTYPE type = DEVPROP_TYPE_EMPTY; DWORD val = 0; DWORD sz = sizeof(val);
		if (!SetupDiGetDevicePropertyW(set, &dev, &key, &type, (PBYTE)&val, sz, &sz, 0)) 
		{
			// if the system has only 1 NUMA node, we should end here with  GetLastError() = ERROR_NOT_FOUND
			return false;
		}
		if (type != DEVPROP_TYPE_UINT32 && type != DEVPROP_TYPE_INT32) 
			return false;
		*out = val; 
		return true;
	}

	// input string example:  "PCI\VEN_1002&DEV_745E&SUBSYS_0E0D1002&REV_CC\6&215957EA&0&00000009"
	// output de SP_DEVINFO_DATA/HDEVINFO of a device
	// returns TRUE if success.
	// Caller must call SetupDiDestroyDeviceInfoList
	bool open_by_instance_id(const std::wstring &instId, HDEVINFO *outSet, SP_DEVINFO_DATA *outDev)
	{
		HDEVINFO set = SetupDiGetClassDevsW(NULL, NULL, NULL, DIGCF_ALLCLASSES);
		if (set == INVALID_HANDLE_VALUE) 
			return false;

		SP_DEVINFO_DATA dev = {}; 
		dev.cbSize = sizeof(dev);
		if (!SetupDiOpenDeviceInfoW(set, instId.c_str(), NULL, 0, &dev)) {
			SetupDiDestroyDeviceInfoList(set); 
			return false;
		}
		*outSet = set; 
		*outDev = dev; 
		return true;
	}




	// the input should be the string from CM_Get_Device_IDW, example: "PCI\VEN_1002&DEV_745E&SUBSYS_0E0D1002&REV_CC\6&215957EA&0&00000009"
	// output is the NUMA index
	// returns TRUE if success.
	bool resolve_numa_from_instance_chain(const std::wstring &startInst, USHORT *outNode)
	{
		std::wstring cur = startInst;

		// walk up the parent chain ( 64 max iterations is just a safety limit ) 
		for (int hop = 0; hop < 64; ++hop) 
		{
			HDEVINFO set = INVALID_HANDLE_VALUE; 
			SP_DEVINFO_DATA dev = {};
			if (!open_by_instance_id(cur, &set, &dev)){
				return false;
			}

			uint32_t node_u32 = 0xFFFFFFFFu;
			if (get_u32_prop(set, dev, DEVPKEY_Device_Numa_Node, &node_u32) && node_u32 != 0xFFFFFFFFu) {
				*outNode = (USHORT)node_u32;
				SetupDiDestroyDeviceInfoList(set);
				return true;
			}

			uint32_t prox = 0;
			if (get_u32_prop(set, dev, DEVPKEY_Numa_Proximity_Domain, &prox)) {
				USHORT n = 0xFFFF;
				if (GetNumaProximityNodeEx((ULONG)prox, &n)) {
					*outNode = n;
					SetupDiDestroyDeviceInfoList(set);
					return true;
				}
			}

			std::wstring parentId;
			bool hasParent = get_str_prop(set, dev, DEVPKEY_Device_Parent, parentId);
			SetupDiDestroyDeviceInfoList(set);
			if (!hasParent) {
				break;
			}
			cur = std::move(parentId);
		}

		return false;
	}


	// description in the header
	int GetNumaNodeForPciBdf(unsigned short seg_in, 
							unsigned char  bus_in, 
							unsigned char  dev_in, 
							unsigned char  func_in, 
							int *out_numa
							)
	{
		if (!out_numa) 
			return -1;

		// set to 0 by default
		*out_numa = 0;

		HDEVINFO set = SetupDiGetClassDevsW(&GUID_DEVCLASS_DISPLAY, NULL, NULL, DIGCF_PRESENT);
		if (set == INVALID_HANDLE_VALUE) {
			return -2;
		}

		for (DWORD i = 0;; ++i) 
		{
			SP_DEVINFO_DATA dev = {}; 
			dev.cbSize = sizeof(dev);
			if (!SetupDiEnumDeviceInfo(set, i, &dev)) 
				break;

			uint32_t seg=0;
			uint32_t bus=0;
			uint32_t dv=0;
			uint32_t fn=0;

			// Microsoft documentation states:  
			//   On single-segment system the string is in this format:             "PCI bus 1, device 2, function 3"
			//   On multi-segment system it looks like this:              "PCI segment 1 bus 2, device 3, function 4"
			// Also this string is 'localizable' so it may be in different languages.
			std::wstring locinfo;
			if (!get_str_prop(set, dev, DEVPKEY_Device_LocationInfo, locinfo))
				continue;


			std::vector<uint32_t> intsList = extract_uints(locinfo);
			if ( intsList.size() == 3 )
			{
				bus = intsList[0];
				dv = intsList[1];
				fn = intsList[2];
			}
			else if ( intsList.size() == 4 )
			{
				seg = intsList[0];
				bus = intsList[1];
				dv = intsList[2];
				fn = intsList[3];
			}
			else
				continue; // error for this device. we expect 3 or 4 ints.


			// check we are on the correct GPU based on the: segment:bus:device.function
			if (seg != seg_in || bus != bus_in || dv != dev_in || fn != func_in) 
				continue;

			// Resolve NUMA from instance chain (stable, not localized).
			ULONG id_sz = 0;
			if (CM_Get_Device_ID_Size(&id_sz, dev.DevInst, 0) != CR_SUCCESS) 
			{ 
				SetupDiDestroyDeviceInfoList(set); 
				return -3; 
			}
			std::vector<wchar_t> id(id_sz + 1);
			if (CM_Get_Device_IDW(dev.DevInst, id.data(), (ULONG)id.size(), 0) != CR_SUCCESS) 
			{ 
				SetupDiDestroyDeviceInfoList(set); 
				return -3; 
			}

			USHORT node = 0xFFFF;
			if (!resolve_numa_from_instance_chain(id.data(), &node)) 
			{ 
				SetupDiDestroyDeviceInfoList(set); 
				return -4; 
			}

			*out_numa = (int)node;
			SetupDiDestroyDeviceInfoList(set);
			return 0;
		}

		SetupDiDestroyDeviceInfoList(set);
		return -5; // not found
	}

	int ParsePciBusId(const char *s, unsigned *domain, unsigned *bus, unsigned *dev, unsigned *func)
	{
		unsigned d = 0, b = 0, de = 0, f = 0;
		if (!s) 
			return -1;
		if (sscanf(s, "%x:%x:%x.%x", &d, &b, &de, &f) != 4) 
		return -1;
		if (domain) *domain = d;
		if (bus) *bus = b;
		if (dev) *dev = de;
		if (func) *func = f;
			return 0;
	}



} // namespace NumaWindowsTool





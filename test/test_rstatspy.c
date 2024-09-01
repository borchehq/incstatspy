#include <stdio.h>
#include <Python.h>


int run_python_script(const char *script_path) {
    int fail = 0;
    FILE* fp = NULL;

    fp = fopen(script_path, "r");
    if (!fp) {
        fail = 1;
        goto except;
    }
     // Ensure current directory is in sys.path
    PyObject *sys_path = PySys_GetObject("path");
    PyObject *current_dir = PyUnicode_FromString(".");
    PyList_Append(sys_path, current_dir);
    Py_DECREF(current_dir);
    
    if(PyRun_SimpleFile(fp, script_path) != 0) {
        fail = 2;
        goto except;
    }

except:
    if(fp) {
        fclose(fp);
    }
    return fail;
}

void test_cpython_code(const char *script_path) {
    wchar_t *program_name = L"TestCPythonExtensions";
    Py_SetProgramName(program_name);  /* optional, recommended */
    Py_Initialize();
    int status = run_python_script(script_path);
    if(status == 1) {
        printf("No such file.\n");
    }
    else if(status == 2) {
        printf("Couldn't execute python script.\n");
    }
    Py_Finalize();
}

int main(int argc, const char *argv[]) {
    printf("Testing...\n");
    test_cpython_code(argv[1]);
    printf("Testing done...\n");
    return 0;
}

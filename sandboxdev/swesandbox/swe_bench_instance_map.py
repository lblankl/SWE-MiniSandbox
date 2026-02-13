instance_to_skip=[
        'django__django-14771',
        'sphinx-doc__sphinx-7985',
        "scikit-learn__scikit-learn-14710",
        'django__django-13837',
        'django__django-14311',
        "sphinx-doc__sphinx-10435",
        "django__django-14792",
        "django__django-13809",
        "astropy__astropy-8872"
        ]
def instance_map(instance_id,repo):
    #scikit-learn__scikit-learn-14983
    if instance_id=='scikit-learn__scikit-learn-14983':
        return [
              'test_shufflesplit_errors[None-train_size3]'
        ]
    if instance_id=='astropy__astropy-7606':
        return ['test_compose_roundtrip[]']
    #if instance_id=="pytest-dev__pytest-10356":
    #    return ['test_mark.py::test_mark_mro','TestKeywordSelection::test_no_match_directories_outside_the_suite']
    if instance_id=='pydata__xarray-6992':
        return ['xarray/tests/test_dataarray.py::TestDataArray::test_to_and_from_cdms2_classic', 
        'xarray/tests/test_dataarray.py::TestDataArray::test_to_and_from_cdms2_ugrid', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_to_and_from_iris', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_to_and_from_iris_dask', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[var_name-height-Height-var_name-attrs0]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[None-height-Height-height-attrs1]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[None-None-Height-Height-attrs2]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[None-None-None-None-attrs3]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[var_name-height-Height-var_name-attrs0]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[None-height-Height-height-attrs1]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[None-None-Height-Height-attrs2]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[None-None-None-unknown-attrs3]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_prevent_duplicate_coord_names', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_fallback_to_iris_AuxCoord[coord_values0]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_fallback_to_iris_AuxCoord[coord_values1]']
    if instance_id=='pydata__xarray-4695':
        return ['xarray/tests/test_dataarray.py::TestDataArray::test_to_and_from_cdms2_classic', 
        'xarray/tests/test_dataarray.py::TestDataArray::test_to_and_from_cdms2_ugrid', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_to_and_from_iris', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_to_and_from_iris_dask', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[var_name-height-Height-var_name-attrs0]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[None-height-Height-height-attrs1]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[None-None-Height-Height-attrs2]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[None-None-None-None-attrs3]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[var_name-height-Height-var_name-attrs0]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[None-height-Height-height-attrs1]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[None-None-Height-Height-attrs2]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[None-None-None-unknown-attrs3]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_prevent_duplicate_coord_names', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_fallback_to_iris_AuxCoord[coord_values0]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_fallback_to_iris_AuxCoord[coord_values1]']
    if instance_id=='pydata__xarray-3305':
        return ['xarray/tests/test_dataarray.py::TestDataArray::test_to_and_from_cdms2_classic', 
        'xarray/tests/test_dataarray.py::TestDataArray::test_to_and_from_cdms2_ugrid', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_to_and_from_iris', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_to_and_from_iris_dask', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[var_name-height-Height-var_name-attrs0]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[None-height-Height-height-attrs1]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[None-None-Height-Height-attrs2]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_name_from_cube[None-None-None-None-attrs3]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[var_name-height-Height-var_name-attrs0]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[None-height-Height-height-attrs1]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[None-None-Height-Height-attrs2]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_da_coord_name_from_cube[None-None-None-unknown-attrs3]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_prevent_duplicate_coord_names', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_fallback_to_iris_AuxCoord[coord_values0]', 
        'xarray/tests/test_dataarray.py::TestIrisConversion::test_fallback_to_iris_AuxCoord[coord_values1]']
    if instance_id=='pydata__xarray-4687':
        return  ['xarray/tests/test_units.py::TestPlots::test_units_in_line_plot_labels[1-coord_attrs0]', 
        'xarray/tests/test_units.py::TestPlots::test_units_in_slice_line_plot_labels_sel[1-coord_attrs0]', 
        'xarray/tests/test_units.py::TestPlots::test_units_in_slice_line_plot_labels_isel[1-coord_attrs0]', 
        'xarray/tests/test_units.py::TestPlots::test_units_in_2d_plot_colorbar_label', 
        'xarray/tests/test_units.py::TestPlots::test_units_facetgrid_plot_labels', 
        'xarray/tests/test_units.py::TestPlots::test_units_facetgrid_2d_imshow_plot_colorbar_labels', 
        'xarray/tests/test_units.py::TestPlots::test_units_facetgrid_2d_contourf_plot_colorbar_labels']
    if repo=="matplotlib/matplotlib":
        return ["test_https_imread_smoketest"]
    if repo=='psf/requests':
        return [
                "test_mixed_case_scheme_acceptable",
                "test_conflicting_post_params",
                "test_pyopenssl_redirect",
                "test_auth_is_stripped_on_redirect_off_host",
                "test_mixed_case_scheme_acceptable",
                "test_requests_history_is_saved",
                "test_stream_timeout"
                    
            ]
    if repo=='sphinx-doc/sphinx':
        return [
                "test_pyfunction_signature_full_py38",
                "test_build_linkcheck.py::test_anchors_ignored",
                "test_build_linkcheck.py::test_defaults_json",
                "test_build_linkcheck.py::test_defaults",
                "test_directive_code.py::test_literal_include_linenos",
                "test_directive_code.py::test_linenothreshold"
        ]
    if repo=='astropy/astropy':
           
        return [
                "test_fitsheader_script","test_fitsheader_table_feature"
            ]
    
    if repo=='astropy/astropy':
          
        return [
                "test_fitsheader_script","test_fitsheader_table_feature"
            ]
    if repo=='pydata/xarray':
           
        return [
                "test_sel_categorical_error",
                "test_categorical_multiindex",
                "test_from_dataframe_categorical",
                "test_sel_categorical",

                'test_variable.py::TestNumpyCoercion::test_from_pint[Variable]', 
                'test_variable.py::TestNumpyCoercion::test_from_pint[IndexVariable]', 
                'test_variable.py::TestNumpyCoercion::test_from_sparse[Variable]', 
                'test_variable.py::TestNumpyCoercion::test_from_pint_wrapping_dask[Variable]', 
                'test_variable.py::TestNumpyCoercion::test_from_pint_wrapping_dask[IndexVariable]'

            ]
    return []
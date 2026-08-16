WORD32 ih264d_parse_sps(dec_struct_t *ps_dec, dec_bit_stream_t *ps_bitstrm)

{
UWORD8 i;
dec_seq_params_t *ps_seq = NULL;
    UWORD8 u1_profile_idc, u1_level_idc, u1_seq_parameter_set_id;
UWORD16 i2_max_frm_num;
UWORD32 *pu4_bitstrm_buf = ps_bitstrm->pu4_buffer;
UWORD32 *pu4_bitstrm_ofst = &ps_bitstrm->u4_ofst;
UWORD8 u1_frm, uc_constraint_set0_flag, uc_constraint_set1_flag;
WORD32 i4_cropped_ht, i4_cropped_wd;
UWORD32 u4_temp;
WORD32 pic_height_in_map_units_minus1 = 0;
UWORD32 u2_pic_wd = 0;
UWORD32 u2_pic_ht = 0;
UWORD32 u2_frm_wd_y = 0;
UWORD32 u2_frm_ht_y = 0;
UWORD32 u2_frm_wd_uv = 0;
UWORD32 u2_frm_ht_uv = 0;
UWORD32 u2_crop_offset_y = 0;
UWORD32 u2_crop_offset_uv = 0;
WORD32 ret;
UWORD32 u4_num_reorder_frames;
/* High profile related syntax element */
WORD32 i4_i;
/* G050 */
UWORD8 u1_frame_cropping_flag, u1_frame_cropping_rect_left_ofst,
u1_frame_cropping_rect_right_ofst,
u1_frame_cropping_rect_top_ofst,
u1_frame_cropping_rect_bottom_ofst;
/* G050 */
/*--------------------------------------------------------------------*/
/* Decode seq_parameter_set_id and profile and level values           */
/*--------------------------------------------------------------------*/
SWITCHONTRACE;
u1_profile_idc = ih264d_get_bits_h264(ps_bitstrm, 8);
COPYTHECONTEXT("SPS: profile_idc",u1_profile_idc);

/* G050 */
uc_constraint_set0_flag = ih264d_get_bit_h264(ps_bitstrm);
uc_constraint_set1_flag = ih264d_get_bit_h264(ps_bitstrm);
ih264d_get_bit_h264(ps_bitstrm);

/*****************************************************/
/* Read 5 bits for uc_constraint_set3_flag (1 bit)   */
/* and reserved_zero_4bits (4 bits) - Sushant        */
/*****************************************************/
ih264d_get_bits_h264(ps_bitstrm, 5);
/* G050 */

/* Check whether particular profile is suported or not */
/* Check whether particular profile is suported or not */
if((u1_profile_idc != MAIN_PROFILE_IDC) &&

(u1_profile_idc != BASE_PROFILE_IDC) &&

(u1_profile_idc != HIGH_PROFILE_IDC)

)
{

/* Apart from Baseline, main and high profile,
* only extended profile is supported provided
* uc_constraint_set0_flag or uc_constraint_set1_flag are set to 1
*/
if((u1_profile_idc != EXTENDED_PROFILE_IDC) ||
((uc_constraint_set1_flag != 1) && (uc_constraint_set0_flag != 1)))
{
return (ERROR_FEATURE_UNAVAIL);
}
}

u1_level_idc = ih264d_get_bits_h264(ps_bitstrm, 8);



COPYTHECONTEXT("SPS: u4_level_idc",u1_level_idc);

u4_temp = ih264d_uev(pu4_bitstrm_ofst, pu4_bitstrm_buf);
if(u4_temp & MASK_ERR_SEQ_SET_ID)
return ERROR_INV_SPS_PPS_T;
u1_seq_parameter_set_id = u4_temp;
COPYTHECONTEXT("SPS: seq_parameter_set_id",
u1_seq_parameter_set_id);

/*--------------------------------------------------------------------*/
/* Find an seq param entry in seqparam array of decStruct             */
/*--------------------------------------------------------------------*/

ps_seq = ps_dec->pv_scratch_sps_pps;
if(ps_dec->i4_header_decoded & 1)
{
*ps_seq = *ps_dec->ps_cur_sps;
}


if((ps_dec->i4_header_decoded & 1) && (ps_seq->u1_profile_idc != u1_profile_idc))
{
ps_dec->u1_res_changed = 1;
return IVD_RES_CHANGED;
}

if((ps_dec->i4_header_decoded & 1) && (ps_seq->u1_level_idc != u1_level_idc))
{
ps_dec->u1_res_changed = 1;
return IVD_RES_CHANGED;
}

ps_seq->u1_profile_idc = u1_profile_idc;
ps_seq->u1_level_idc = u1_level_idc;
ps_seq->u1_seq_parameter_set_id = u1_seq_parameter_set_id;

/*******************************************************************/
/* Initializations for high profile - Sushant                      */
/*******************************************************************/
ps_seq->i4_chroma_format_idc = 1;
ps_seq->i4_bit_depth_luma_minus8 = 0;
ps_seq->i4_bit_depth_chroma_minus8 = 0;
ps_seq->i4_qpprime_y_zero_transform_bypass_flag = 0;
ps_seq->i4_seq_scaling_matrix_present_flag = 0;
if(u1_profile_idc == HIGH_PROFILE_IDC)
{

/* reading chroma_format_idc   */
ps_seq->i4_chroma_format_idc = ih264d_uev(pu4_bitstrm_ofst,
pu4_bitstrm_buf);

/* Monochrome is not supported */
if(ps_seq->i4_chroma_format_idc != 1)
{
return ERROR_INV_SPS_PPS_T;
}

/* reading bit_depth_luma_minus8   */
ps_seq->i4_bit_depth_luma_minus8 = ih264d_uev(pu4_bitstrm_ofst,
pu4_bitstrm_buf);

if(ps_seq->i4_bit_depth_luma_minus8 != 0)
{
return ERROR_INV_SPS_PPS_T;
}

/* reading bit_depth_chroma_minus8   */
ps_seq->i4_bit_depth_chroma_minus8 = ih264d_uev(pu4_bitstrm_ofst,
pu4_bitstrm_buf);

if(ps_seq->i4_bit_depth_chroma_minus8 != 0)
{
return ERROR_INV_SPS_PPS_T;
}

/* reading qpprime_y_zero_transform_bypass_flag   */
ps_seq->i4_qpprime_y_zero_transform_bypass_flag =
(WORD32)ih264d_get_bit_h264(ps_bitstrm);

if(ps_seq->i4_qpprime_y_zero_transform_bypass_flag != 0)
{
return ERROR_INV_SPS_PPS_T;
}

/* reading seq_scaling_matrix_present_flag   */
ps_seq->i4_seq_scaling_matrix_present_flag =
(WORD32)ih264d_get_bit_h264(ps_bitstrm);

if(ps_seq->i4_seq_scaling_matrix_present_flag)
{
for(i4_i = 0; i4_i < 8; i4_i++)
{
ps_seq->u1_seq_scaling_list_present_flag[i4_i] =
ih264d_get_bit_h264(ps_bitstrm);

/* initialize u1_use_default_scaling_matrix_flag[i4_i] to zero */
/* before calling scaling list                             */
ps_seq->u1_use_default_scaling_matrix_flag[i4_i] = 0;

if(ps_seq->u1_seq_scaling_list_present_flag[i4_i])
{
if(i4_i < 6)
{
ih264d_scaling_list(
ps_seq->i2_scalinglist4x4[i4_i],
16,
&ps_seq->u1_use_default_scaling_matrix_flag[i4_i],
ps_bitstrm);
}
else
{
ih264d_scaling_list(
ps_seq->i2_scalinglist8x8[i4_i - 6],
64,
&ps_seq->u1_use_default_scaling_matrix_flag[i4_i],
ps_bitstrm);
}
}
}
}
}
/*--------------------------------------------------------------------*/
/* Decode MaxFrameNum                                                 */
/*--------------------------------------------------------------------*/
u4_temp = 4 + ih264d_uev(pu4_bitstrm_ofst, pu4_bitstrm_buf);
if(u4_temp > MAX_BITS_IN_FRAME_NUM)
{
return ERROR_INV_SPS_PPS_T;
}
ps_seq->u1_bits_in_frm_num = u4_temp;
COPYTHECONTEXT("SPS: log2_max_frame_num_minus4",
(ps_seq->u1_bits_in_frm_num - 4));

i2_max_frm_num = (1 << (ps_seq->u1_bits_in_frm_num));
ps_seq->u2_u4_max_pic_num_minus1 = i2_max_frm_num - 1;
/*--------------------------------------------------------------------*/
/* Decode picture order count and related values                      */
/*--------------------------------------------------------------------*/
u4_temp = ih264d_uev(pu4_bitstrm_ofst, pu4_bitstrm_buf);

if(u4_temp > MAX_PIC_ORDER_CNT_TYPE)
{
return ERROR_INV_POC_TYPE_T;
}
ps_seq->u1_pic_order_cnt_type = u4_temp;
COPYTHECONTEXT("SPS: pic_order_cnt_type",ps_seq->u1_pic_order_cnt_type);

ps_seq->u1_num_ref_frames_in_pic_order_cnt_cycle = 1;
if(ps_seq->u1_pic_order_cnt_type == 0)
{
u4_temp = 4 + ih264d_uev(pu4_bitstrm_ofst, pu4_bitstrm_buf);
if(u4_temp > MAX_BITS_IN_POC_LSB)
{
return ERROR_INV_SPS_PPS_T;
}
ps_seq->u1_log2_max_pic_order_cnt_lsb_minus = u4_temp;
ps_seq->i4_max_pic_order_cntLsb = (1 << u4_temp);
COPYTHECONTEXT("SPS: log2_max_pic_order_cnt_lsb_minus4",(u4_temp - 4));
}
else if(ps_seq->u1_pic_order_cnt_type == 1)
{
ps_seq->u1_delta_pic_order_always_zero_flag = ih264d_get_bit_h264(
ps_bitstrm);
COPYTHECONTEXT("SPS: delta_pic_order_always_zero_flag",
ps_seq->u1_delta_pic_order_always_zero_flag);

ps_seq->i4_ofst_for_non_ref_pic = ih264d_sev(pu4_bitstrm_ofst,
pu4_bitstrm_buf);
COPYTHECONTEXT("SPS: offset_for_non_ref_pic",
ps_seq->i4_ofst_for_non_ref_pic);

ps_seq->i4_ofst_for_top_to_bottom_field = ih264d_sev(
pu4_bitstrm_ofst, pu4_bitstrm_buf);
COPYTHECONTEXT("SPS: offset_for_top_to_bottom_field",
ps_seq->i4_ofst_for_top_to_bottom_field);

u4_temp = ih264d_uev(pu4_bitstrm_ofst, pu4_bitstrm_buf);
if(u4_temp > 255)
return ERROR_INV_SPS_PPS_T;
ps_seq->u1_num_ref_frames_in_pic_order_cnt_cycle = u4_temp;
COPYTHECONTEXT("SPS: num_ref_frames_in_pic_order_cnt_cycle",
ps_seq->u1_num_ref_frames_in_pic_order_cnt_cycle);

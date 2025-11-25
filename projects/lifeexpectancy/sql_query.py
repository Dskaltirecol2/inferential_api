QUERY_PREDICCION = """
WITH ranked_inspecciones AS (
    SELECT 
        t_inner.flota_kal,
        t_inner.reporte_kal,
        t_inner.hora,
        t_inner.fecha_fin,
        t_inner.horafin,
        t_inner.equipo_kal,
        i.refe, 
        i.pos, 
        i.rtdsfext, 
        i.rtdsfint, 
        i.ajuste, 
        i.nrointerno,
        i.horastoms,
        i.kmtoms,
        i.serialtoms,
        i.nrosf,
        i.hinspec,
        i.codigop,
        desechos.fecha_retiro,
        desechos.razon_r,
        desechos.componente,
        i.pitcrew,
        ROW_NUMBER() OVER (PARTITION BY i.nrosf ORDER BY i.fechains ASC) as rn,
        desechos.horas_logra,
        desechos.distan_logra
    FROM inspecciones i
    INNER JOIN trabajos t_inner ON t_inner.refe = i.refe
    LEFT JOIN acta desechos ON desechos.serie = i.nrosf
    WHERE i.nrointerno = :id_llanta
        AND i.nrointerno IS NOT NULL
        AND UPPER(TRIM(i.nrointerno)) NOT IN (
            'N/A', 'NO TIENE', 'NULL', 'NONE', 'SIN ASIGNAR', 'SIN SERIE',
            'NO', 'NO REGISTRA', 'NO VISIBLE', '-', '--', '---', '----', '*'
        )
        AND TRIM(i.nrointerno) != ''
        AND i.fechains IS NOT NULL
        AND t_inner.cliente = 'Glencore'
        AND t_inner.estado_kal = 'Finalizado'
        AND i.nrosf IS NOT NULL
        AND i.nrosf != ''
        AND desechos.fecha_retiro IS NULL
        AND i.codigop IN ('2','3') 
        AND i.hinspec != 'IF-18'
        AND t_inner.flota_kal IN ('CAT 793','CAT 789','KOM 930E-4','HIT EH5000','HIT EH-4000')
),
primer_inspeccion_con_hallazgo AS (
    SELECT * FROM ranked_inspecciones ri WHERE ri.rn = 1
),
summary_detections AS ( 
    SELECT 
        ri.nrosf,
        SUM(CASE WHEN ri.hinspec = 'IF-23' THEN 1 ELSE 0 END) as baja_presion,
        SUM(CASE WHEN ri.hinspec = 'IF-08' THEN 1 ELSE 0 END) as dano_corte,
        SUM(CASE WHEN ri.hinspec = 'IF-09' THEN 1 ELSE 0 END) as dano_separacion_corte,
        SUM(CASE WHEN ri.hinspec = 'IF-08-01' THEN 1 ELSE 0 END) as dano_corte_costado,
        SUM(CASE WHEN ri.hinspec = 'IF-08-02' THEN 1 ELSE 0 END) as dano_corte_banda,
        SUM(CASE WHEN ri.hinspec = 'IF-42' THEN 1 ELSE 0 END) as desgaste_irregular,
        SUM(CASE WHEN ri.hinspec = 'IF-08-03' THEN 1 ELSE 0 END) as dano_corte_hombro,
        SUM(CASE WHEN ri.hinspec = 'IF-30' THEN 1 ELSE 0 END) as falla_reparacion_reencauchaje,
        SUM(CASE WHEN ri.hinspec = 'IF-48' THEN 1 ELSE 0 END) as desgaste_total,
        SUM(CASE WHEN ri.hinspec = 'IF-08-04' THEN 1 ELSE 0 END) as dano_corte_arrancabanda,
        SUM(CASE WHEN ri.hinspec = 'IF-03' THEN 1 ELSE 0 END) as carcasa_danada_banda,
        SUM(CASE WHEN ri.hinspec = 'IF-09-01' THEN 1 ELSE 0 END) as dano_separacioncorte_banda,
        SUM(CASE WHEN ri.hinspec = 'IF-38' THEN 1 ELSE 0 END) as burbuja_llanta,
        SUM(CASE WHEN ri.hinspec = 'IF-07' THEN 1 ELSE 0 END) as dano_talon,
        SUM(CASE WHEN ri.hinspec = 'IF-36' THEN 1 ELSE 0 END) as fuga_pequena,
        SUM(CASE WHEN ri.hinspec = 'IF-02' THEN 1 ELSE 0 END) as carcasa_danada_costado,
        SUM(CASE WHEN ri.hinspec = 'IF-19' THEN 1 ELSE 0 END) as llanta_alta_temp,
        SUM(CASE WHEN ri.hinspec = 'IF-01' THEN 1 ELSE 0 END) as carcasada_danada_hombro,
        SUM(CASE WHEN ri.hinspec = 'IF-10' THEN 1 ELSE 0 END) as dano_separacion_calor,
        SUM(CASE WHEN ri.hinspec = 'IF-11' THEN 1 ELSE 0 END) as dano_impacto,
        SUM(CASE WHEN ri.hinspec = 'IF-06' THEN 1 ELSE 0 END) as dano_accidente,
        SUM(CASE WHEN ri.hinspec = 'IF-12' THEN 1 ELSE 0 END) as dano_rayo_ele_fue,
        SUM(CASE WHEN ri.hinspec = 'IF-13' THEN 1 ELSE 0 END) as dano_mecanico,
        SUM(CASE WHEN ri.hinspec = 'IF-14' THEN 1 ELSE 0 END) as dano_sepa_meca,
        SUM(CASE WHEN ri.hinspec = 'IF-20' THEN 1 ELSE 0 END) as impacto,
        SUM(CASE WHEN ri.hinspec = 'IF-32' THEN 1 ELSE 0 END) as penetracion_roca_metal,
        SUM(CASE WHEN ri.hinspec = 'IF-34' THEN 1 ELSE 0 END) as desinflado_rodado,
        SUM(CASE WHEN ri.hinspec = 'IF-37' THEN 1 ELSE 0 END) as cortes_circunferenciales,
        SUM(CASE WHEN ri.hinspec = 'IF-09-02' THEN 1 ELSE 0 END) as dano_sepacor_costado,
        SUM(CASE WHEN ri.hinspec = 'IF-09-03' THEN 1 ELSE 0 END) as dano_sepacor_hombro,
        SUM(CASE WHEN ri.hinspec = 'IF-23-01' THEN 1 ELSE 0 END) as baja_presion_pinchazo,
        SUM(CASE WHEN ri.codigop = '4' THEN 1 ELSE 0 END) as prio_4,
        SUM(CASE WHEN ri.codigop = '3' THEN 1 ELSE 0 END) as prio_3,
        SUM(CASE WHEN ri.codigop = '2' THEN 1 ELSE 0 END) as prio_2,
        SUM(CASE WHEN ri.codigop = '1' THEN 1 ELSE 0 END) as prio_1,
        SUM(CASE WHEN ri.pitcrew = 'SI' THEN 1 ELSE 0 END) as no_pitcrew,
        SUM(CASE WHEN ri.pitcrew = 'SI' AND ri.codigop = '2' THEN 1 ELSE 0 END) as no_pitcrew_2,
        COUNT(*) as total
    FROM ranked_inspecciones ri 
    LEFT JOIN primer_inspeccion_con_hallazgo ich ON ich.nrointerno = ri.nrointerno 
    WHERE ri.hinspec IN ('IF-01','IF-02','IF-03','IF-06','IF-07','IF-08','IF-09','IF-10','IF-11','IF-12','IF-13','IF-14','IF-19','IF-20','IF-23','IF-30','IF-32','IF-34','IF-36','IF-37','IF-38','IF-42','IF-48','IF-08-01','IF-08-03','IF-08-04','IF-09-01','IF-09-02','IF-09-03','IF-23-01')
    AND ri.fecha_fin <= ich.fecha_fin 
    GROUP BY ri.nrosf
),
summary_repa AS (
    SELECT 
        pr.serie,
        AVG(pr.longitud) as longi_prom,
        MIN(pr.longitud) as min_longi,
        MAX(pr.longitud) as max_longi,
        SUM(pr.longitud) as sum_longi,
        AVG(pr.profundidad) as prof_prom,
        MIN(pr.profundidad) as min_prof,
        MAX(pr.profundidad) as max_prof,
        SUM(pr.profundidad) as sum_prof,
        SUM(CASE WHEN pr.tipo = 'Preventive' THEN 1 ELSE 0 END) as no_preventivos,
        SUM(CASE WHEN pr.tipo = 'Corrective' THEN 1 ELSE 0 END) as no_correctivos,
        SUM(CASE WHEN pm.clase = 'Parche' THEN 1 ELSE 0 END) as no_parches,
        SUM(CASE WHEN pm.clase = 'Malla' THEN 1 ELSE 0 END) as no_malla
    FROM produc_repa pr 
    LEFT JOIN reparaciones r ON pr.refe = r.refe
    LEFT JOIN parches_mallas pm ON pm.nro_reparacion = pr.id_drep
    LEFT JOIN primer_inspeccion_con_hallazgo ich ON ich.nrointerno = pr.serie 
    WHERE r.cliente = 'Glencore'
    AND pr.serie = :id_llanta
    AND r.fecha <= ich.fecha_fin
    GROUP BY pr.serie
),
montaje AS (
    SELECT 
        t_inner.equipo_kal, 
        pa.nro_seriefins, 
        pa.nro_serieins, 
        pa.fechamontaje,
        ROW_NUMBER() OVER (PARTITION BY pa.nro_serieins, t_inner.equipo_kal ORDER BY pa.fechamontaje ASC) as rn,
        pa.rtd_ext, 
        pa.rtd_int 
    FROM p_atendidas pa
    INNER JOIN trabajos t_inner ON t_inner.refe = pa.refe
    WHERE t_inner.cliente = 'Glencore'
    AND t_inner.modulo = 'realtime'
    AND pa.nro_seriefins != ''
    AND pa.nro_seriefins IS NOT NULL
    AND pa.nro_serieins = :id_llanta
    AND t_inner.flota_kal IN ('CAT 793','CAT 789','KOM 930E-4','HIT EH5000','HIT EH-4000')
),
ranked_montajes AS (
    SELECT 
        m.equipo_kal, 
        m.nro_seriefins, 
        m.fechamontaje,
        m.nro_serieins,
        m.rtd_ext,
        m.rtd_int
    FROM montaje m WHERE rn = 1
)
SELECT 
    ri.refe as id_inspeccion,
    ri.flota_kal as modelo_flota,
    ri.equipo_kal as id_equipo,
    ri.nrointerno as id_llanta,
    ri.nrosf as serialtoms,
    ri.componente as componente,
    ri.pos as posicion,
    m.fechamontaje as fecha_montaje, 
    ri.reporte_kal as fecha_inicio_inspeccion,
    ri.hora as hora_inicio_inspeccion,
    ri.fecha_fin as fecha_penultima_insp,
    ri.horafin as hora_fin_inspeccion,
    ri.fecha_retiro,
    ri.horastoms as total_horas_llanta,
    ri.kmtoms as total_kms_llanta,
    m.rtd_ext as rtdext_montaje,
    ri.rtdsfext,
    m.rtd_int as rtdint_montaje,
    ri.rtdsfint, 
    ri.ajuste,
    sd.baja_presion as no_baja_presion,
    sd.dano_corte as no_dano_corte,
    sd.dano_separacion_corte,
    sd.dano_corte_costado,
    sd.dano_corte_banda,
    sd.desgaste_irregular,
    sd.dano_corte_hombro,
    sd.falla_reparacion_reencauchaje,
    sd.desgaste_total,
    sd.dano_corte_arrancabanda,
    sd.carcasa_danada_banda,
    sd.dano_separacioncorte_banda,
    sd.burbuja_llanta,
    sd.dano_talon,
    sd.fuga_pequena,
    sd.carcasa_danada_costado,
    sd.llanta_alta_temp,
    sd.carcasada_danada_hombro,
    sd.dano_separacion_calor,
    sd.dano_impacto,
    sd.dano_accidente,
    sd.dano_rayo_ele_fue,
    sd.dano_mecanico,
    sd.dano_sepa_meca,
    sd.impacto,
    sd.penetracion_roca_metal,
    sd.desinflado_rodado,
    sd.cortes_circunferenciales,
    sd.dano_sepacor_costado,
    sd.dano_sepacor_hombro,
    sd.baja_presion_pinchazo,
    sd.prio_3,
    sd.prio_2,
    sd.prio_1,
    sr.max_longi,
    sr.max_prof,
    sr.min_longi,
    sr.min_prof,
    sr.prof_prom,
    sr.sum_prof,
    sr.sum_longi,
    sr.longi_prom,
    sr.no_correctivos,
    sr.no_preventivos,
    sr.no_parches,
    sr.no_malla,
    sd.no_pitcrew,
    sd.no_pitcrew_2,
    DATEDIFF(ri.fecha_fin, m.fechamontaje) as diasprimerfalla
FROM primer_inspeccion_con_hallazgo ri 
LEFT JOIN summary_detections sd ON sd.nrosf = ri.nrosf
LEFT JOIN summary_repa sr ON sr.serie = ri.nrointerno
LEFT JOIN ranked_montajes m ON m.equipo_kal = ri.equipo_kal AND m.nro_serieins = ri.nrointerno   
WHERE ri.rn = 1  
    AND ri.fecha_fin IS NOT NULL
    AND m.fechamontaje >= '2024-01-01'
    AND DATEDIFF(ri.fecha_fin, m.fechamontaje) > 0
ORDER BY ri.fecha_fin DESC, ri.horafin DESC
LIMIT 1;
"""
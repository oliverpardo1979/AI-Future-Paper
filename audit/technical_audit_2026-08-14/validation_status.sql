SELECT status, count
FROM (
    SELECT 1 AS sort_order, 'Respaldado en su alcance' AS status, 8 AS count
    UNION ALL
    SELECT 2, 'Parcial o condicional', 2
    UNION ALL
    SELECT 3, 'No respaldado', 1
    UNION ALL
    SELECT 4, 'Contradicho', 1
)
ORDER BY sort_order;

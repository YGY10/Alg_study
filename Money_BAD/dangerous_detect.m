function dangerous_detect()
    clf; hold on; grid on; axis equal;

    %===== 测试两车参数 =====%
    ego_x = 0; 
    ego_y = 5;
    ego_heading = deg2rad(0);   % 朝右
    ego_v = 1;

    obs_x = 20;
    obs_y = 8;
    obs_heading = deg2rad(90); % 朝左
    obs_v = 2;

    %===== 计算碰撞点 =====%
    [is_danger, cx, cy] = fcn( ...
        ego_x, ego_y, ego_heading, ego_v, ...
        obs_x, obs_y, obs_heading, obs_v);

    if is_danger
        fprintf("⚠️ 发生潜在碰撞！碰撞点: (%.2f, %.2f)\n", cx, cy);
    else
        fprintf("✅ 无碰撞风险\n");
    end

    %===== 绘制未来轨迹（直线 0~5s）=====%
    T = linspace(0,5,50);

    ego_traj = [ego_x + ego_v*cos(ego_heading).*T;
                ego_y + ego_v*sin(ego_heading).*T];

    obs_traj = [obs_x + obs_v*cos(obs_heading).*T;
                obs_y + obs_v*sin(obs_heading).*T];

    plot(ego_traj(1,:), ego_traj(2,:), 'b-', 'LineWidth', 2);
    plot(obs_traj(1,:), obs_traj(2,:), 'r-', 'LineWidth', 2);

    scatter(ego_x, ego_y, 60, 'b', 'filled');
    scatter(obs_x, obs_y, 60, 'r', 'filled');

    %===== 如果有碰撞点，绘制出来 =====%
    if is_danger
        scatter(cx, cy, 100, 'm', 'filled');
        text(cx, cy, '   Collision!', 'Color','magenta', 'FontSize',12);
    end

    legend("ego future", "obs future", "ego start", "obs start", "collision point");
    title("DD");
    uiwait();
end







function [is_danger, col_x, col_y] = fcn( ...
    x1, y1, h1, v1, ...
    x2, y2, h2, v2)

    P1 = [x1, y1];
    P2 = [x2, y2];

    D1 = [v1*cos(h1), v1*sin(h1)];
    D2 = [v2*cos(h2), v2*sin(h2)];

    cross2 = @(a,b) a(1)*b(2)-a(2)*b(1);
    dot2 = @(a,b) a(1)*b(1)+a(2)*b(2);

    r = P2 - P1;

    denom = cross2(D1, D2);

    is_danger = false;
    col_x = NaN;
    col_y = NaN;

    %=====================
    % 1) 平行 / 反向
    %=====================
    if abs(denom) < 1e-8

        % 不共线 → 不可能碰撞
        if abs(cross2(r, D1)) > 1e-8
            return;
        end

        v_rel = D2 - D1;
        vrm = dot2(v_rel, v_rel);

        if vrm < 1e-12
            return;
        end

        t = - dot2(r, v_rel) / vrm;

        if t >= 0
            C1 = P1 + D1 * t;
            C2 = P2 + D2 * t;
            C = 0.5*(C1 + C2);
            is_danger = true;
            col_x = C(1);
            col_y = C(2);
        end
        return;
    end

    %=====================
    % 2) 非平行，正常求交点
    %=====================
    t1 = cross2(r, D2)/denom;
    t2 = cross2(r, D1)/denom;

    if t1 >= 0 && t2 >= 0
        C = P1 + t1*D1;
        is_danger = true;
        col_x = C(1);
        col_y = C(2);
    end
end


